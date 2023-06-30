from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import AutoTokenizer
from torch import cuda
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline

from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForChainRun
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

lora_config = LoraConfig.from_pretrained("AyoubMDL/private_doctor")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("AyoubMDL/private_doctor")
model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map={"": 0})

model = get_peft_model(model, lora_config)

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x) for x in [
        ['###', 'Human', ':'], ['User', ':']
    ]
]

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=256,  # max number of tokens to generate in the output
    repetition_penalty=1.2  # without this output begins repeating
)


# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are a helpful AI assistant in the medical field, you will answer the users query
            with a short but precise answer. If you are not sure about the answer you state
            "I don't know". This is a conversation, not a webpage, there should be ZERO HTML
            in the response.

            Remember, Assistant responses are short. Here is the conversation:

            ### Human: {query}
            ### Assistant: """
    )

llm = HuggingFacePipeline(pipeline=generate_text)


class DoctorFalconChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    output_key: str = "text"
    suffixes = ['', 'User:', 'User ', 'system:', 'Assistant:']

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # format the prompt
        prompt_value = self.prompt.format_prompt(**inputs)
        # generate response from llm
        response = self.llm.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )
        # _______________
        # here we add the removesuffix logic
        for suffix in self.suffixes:
            response.generations[0][0].text = response.generations[0][0].text.removesuffix(
                suffix)

        return {self.output_key: response.generations[0][0].text.lstrip()}

    async def _acall(
        self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async is not supported for this chain.")

    @property
    def _chain_type(self) -> str:
        return "open_llama_chat_chain"

    def predict(self, query: str) -> str:
        out = self._call(inputs={'query': query})
        return out['text']


falcon_chain = DoctorFalconChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    text = str(input("Ask the doctor : "))
    output = falcon_chain.predict(
        query=text
    )
    print(text)
