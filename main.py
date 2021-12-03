import pandas as pd
from config import *
import streamlit as st
import ner_predict as nerp
from st_aggrid import AgGrid

class CQAbot():

    def get_data(self, path):
        return pd.read_csv(path).applymap(str)

    def calc_agg_operation(self, predicted_agg, answers):
        if predicted_agg == 'SUM':
            answer_split = answers.split(',')
            numbers_list = list(map(lambda x: float(x.strip()), answer_split))
            return str(sum(numbers_list))
        
        if predicted_agg == 'COUNT':
            answer_split = answers.split(',')
            return str(len(answer_split))
        
        if predicted_agg == 'AVERAGE':
            answer_split = answers.split(',')
            numbers_list = list(map(lambda x: float(x.strip()), answer_split))
            return str(sum(numbers_list)/len(numbers_list))

    def get_data(self, path):
        return pd.read_csv(path).round(3).applymap(str)

    def convert_to_dataframe(self, df):
        return pd.DataFrame.from_dict(df.to_dict(orient='list'))

    def get_model_prediction(self, df):
        self.table = self.convert_to_dataframe(df)
        input = tokenizer(table=self.table, queries=[question], padding='max_length', return_tensors='pt', truncation=True)
        output = model(**input)

        return tokenizer.convert_logits_to_predictions(
            input,
            output.logits.detach(),
            output.logits_aggregation.detach()
        )

    def get_prediction_text(self, queries, predicted_answer_position, aggregation_predictions):
        answers = []

        for coordinates in predicted_answer_position:

            if len(coordinates) == 1:
                answers.append(self.table.iat[coordinates[0]])

            else:
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(self.table.iat[coordinate])

                answers.append(", ".join(cell_values))

        for query, answers, predicted_agg in zip(queries, answers, aggregation_predictions):
            print(query)

            def validate_numeric(answer_list):
                for answer in answer_list:
                    # print(answer)
                    try:
                        float(answer)
                    except:
                        return False
                return True
            
            answer_list = list(map(str.strip, answers.split(',')))
            
            is_input_digit = validate_numeric(answer_list)

            if len(answers.strip()) == 0:   
                return 'No data to display'

            elif predicted_agg == "NONE":
                print("Prediction 0: " + answers)
                return "Prediction: " + answers

            elif is_input_digit == False:
                print("Prediction 1: " + answers)
                # return "Prediction: " + answers
                return pd.DataFrame(answer_list, columns=['Result'])

            elif is_input_digit == True:
                result_agg = self.calc_agg_operation(predicted_agg, answers)
                print("Prediction 2: " + predicted_agg + " > " + result_agg)
                # return "Prediction: " + result_agg
                return pd.DataFrame([result_agg], columns=['Result'])

def header_text(value):
    st.markdown("<h3 style='font-family:Georgia; font-size:25px;'>{0}</h3>".format(value), unsafe_allow_html=True)


if __name__ == "__main__":
    cqa = CQAbot()

    st.markdown("<p style='font-family:Georgia; font-size:50px;'>CleverQA Bot</p>", unsafe_allow_html=True)
    header_text('1) Original dataset')
    AgGrid(cqa.get_data(data_path), fit_columns_on_grid_load=True)

    question = st.text_area("")

    if st.button('Tell me'):

        queries = [question]

        df = nerp.ner_filter(cqa.get_data(data_path), question)
        
        header_text('3) Filtering significant data points')
        AgGrid(df, fit_columns_on_grid_load=True, height=200)

        predicted_answer_position, predicted_aggregation_indices = cqa.get_model_prediction(df)


        aggregation_predictions = [aggregations[x] for x in predicted_aggregation_indices]

        prediction = cqa.get_prediction_text([question], predicted_answer_position, aggregation_predictions)

        st.title('Result')
        st.write(prediction)




