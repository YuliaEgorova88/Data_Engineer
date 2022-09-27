import logging
from datetime import datetime
import pandas as pd
import os
import json
import dill


path = os.environ.get('PROJECT_PATH', '.')


def get_test():
    test_examples = []
    for filename in os.listdir(f'{path}/../data/test/'):
        with open(f'{path}/../data/test/' + filename, 'rb') as file:
            example = json.load(file)
            test_examples.append(example)

    return test_examples

    # pkl_filename = f'{path}/../data/models/cars_pipe_202209271510.pkl'
    # with open(pkl_filename, 'rb') as file:
    #     model = dill.load(file)
    # return model
    #


def predict():
    model_name = list(os.listdir(f'{path}/../data/models'))[0]
    with open(f'{path}/../data/models/{model_name}', 'rb') as file:
        best_model = dill.load(file)

    test_examples = get_test()

    df = pd.DataFrame.from_dict(test_examples)
    df['pred'] = best_model.predict(df)

    logging.info(df[['id', 'pred']])

    predict_name = f'{path}/../data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df[['id', 'pred']].to_csv(predict_name, index=False)

    logging.info(f'Predict is saved as {predict_name}')
    #
    #
    # model = read_pkl()
    # pred = {}
    #
    # for i in os.listdir(f'{path}/../data/test'):
    #     with open(f'{path}/../data/test/{i}', 'rb') as f:
    #         data = json.load(f)
    #     df = pd.DataFrame([data])
    #     age_category = model.predict(df)
    #     pred[df.loc[0, 'id']] = age_category
    # df_2 = pd.DataFrame(list(pred.items()),
    #                     columns=['car_id', 'pred'])
    # df_2.to_csv(f'{path}/../data/predictions/preds_202207191811.csv')


if __name__ == '__main__':
    predict()
