# DoctorFalcon

DoctorFalcon is a fine-tuned version of falcon-7b-instruct using the baize medical dataset. It incorporates the Qlora technique for fine-tuning the model. The repository consists of the following files:

- `train.py`: This file is used to fine-tune the model using the baize medical dataset and the Qlora technique.
- `predict.py`: This file allows you to test the fine-tuned model by making predictions.
- `app.py`: This file contains a Streamlit web app that provides an interactive interface to interact with the fine-tuned model.

## Usage

To use DoctorFalcon, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/DoctorFalcon.git
   ```
2. Install the required dependencies:
```shell
   pip install -r requirements.txt
```
3. Fine-tune the model:
```shell
   python train.py
```
4. Test the model:
```shell
   python predict.py
```
5. Run the Streamlit web app:
```shell
   streamlit run app.py
```
This will start the Streamlit web app, allowing you to interact with the fine-tuned model through a user-friendly interface.

Feel free to explore the code and make any necessary modifications to suit your needs.
