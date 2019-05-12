from data import input_fn
import numpy as np

def pprint_result(result):
  print("\nResults:")
  for key, val in result.items():
    print(f"  {key}: {val}")
  print()

def peek_classification(estimator, test_df):
  predict_df = test_df[150:200].copy()

  pred_iter = estimator.predict(lambda:input_fn(predict_df, num_epochs=1, shuffle=False, batch_size=10))

  classes = np.array(['outside', 'tram'])
  pred_class_id = []

  for pred_dict in pred_iter:
    pred_class_id.append(pred_dict['class_ids'])

  predict_df['predicted_class'] = classes[np.array(pred_class_id)]
  predict_df['correct'] = (predict_df['predicted_class'] == 'tram') == predict_df['imInTram']

  print(predict_df[['imInTram','predicted_class', 'correct']])
