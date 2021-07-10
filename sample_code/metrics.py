import json
import os
import tensorflow as tf


class Metrics:
    
    def __init__(self, label_list, eval_dir):
        self.label_list = label_list
        self.eval_dir = eval_dir
        
    def json_writer(self, inputs, labels, predictions, file_name):
        predictions_args = tf.math.argmax(predictions, 1)
        scores = tf.math.reduce_max(predictions, axis=1)
        answers = labels.numpy()
        preds = predictions_args.numpy()
    
        confusion_matrix = tf.math.confusion_matrix(answers, preds, num_classes=len(self.label_list)).numpy()
        total = len(predictions)
        evaluation_result = {
            "summary": {
                "total": total,
                "labels": self.label_list,
                "metrics": []
            },
            "results": []
        }
        for i in range(total):
            result = {
                "input": "An image",
                "answer": int(answers[i]),
                "pred": int(preds[i]),
                "score": float(scores[i]),
                "is_correct": bool(answers[i] == preds[i])
            }
            evaluation_result["results"].append(result)
    

        for i, label in enumerate(self.label_list):
            TP, FP, FN, TN = 0, 0, 0, 0
            answer_count, pred_count = 0, 0

            for j in range(len(self.label_list)):
                for k in range(len(self.label_list)):
                    cell_value = int(confusion_matrix[j][k])
                    if i == j:
                        answer_count += cell_value
                        if i == k:
                            pred_count += cell_value
                            TP += cell_value
                        else:
                            FN += cell_value
                    else:
                        if i == k:
                            pred_count += cell_value
                            FP += cell_value
                        else:
                            TN += cell_value
            precision = TP / pred_count
            recall = TP / answer_count
            metric = {
                "label": label,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "answer_count": answer_count,
                "pred_count": pred_count,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
            }
            evaluation_result["summary"]["metrics"].append(metric)

            print(f'Label: {label}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')
            print(f'Prediction: {pred_count}')
            print(f'Answer: {answer_count}')
        
        total_metric = {}
        for key in ['precision', 'recall', 'f1']:
            total = 0
            for metric in evaluation_result["summary"]["metrics"]:
                total += metric.get(key)
            total_metric.update({
                key: total / len(self.label_list)
            })
        evaluation_result["summary"].update(total_metric)
        print(total_metric)
    
        if not os.path.isdir(self.eval_dir):
            os.mkdir(self.eval_dir)
        
        result_json = os.path.join(self.eval_dir, file_name)
        with open(result_json, 'w') as result_file:
            json.dump(evaluation_result, result_file)
    
        print('Save evaluation result success.')
