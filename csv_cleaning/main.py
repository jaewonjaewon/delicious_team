import csv
import json
import os
from io import StringIO
from datetime import datetime
from typing import Callable
from tqdm import tqdm


class Encoding:
    productcodes = {}
    devicetypes = {}
    alphabets = {
        "X": 0,
        "O": 1,
        "N": 2,
        "Y": 3,
        "F": 4,
        "M": 5,
        "V": 6,
        "L": 7,
        "E": 8
    }

    def __init__(self):
        with open('productcode.json') as json_file:
            productcode_json: list = json.load(json_file)
            for productcode in productcode_json:
                self.productcodes[productcode['productcode']] = productcode['encoding']
        with open('devicetype.json') as json_file:
            devicetype_json: list = json.load(json_file)
            for devicetype in devicetype_json:
                self.devicetypes[devicetype['devicetype']] = devicetype['encoding']

    @staticmethod
    def timestamp(x):
        return datetime.fromisoformat(x).timestamp()

    def product_encoding(self, value):
        if not value:
            return 0
        return self.productcodes.get(value, 0)

    def device_encoding(self, value):
        if not value:
            return 0
        return self.devicetypes.get(value, 0)

    def alphabet_encoding(self, value):
        if not value:
            return 7
        return self.alphabets.get(value, 7)

    def get_encodings(self):
        encodings = {
            "uno": {'type': str, 'default': 0},
            "registerdate": {'type': self.timestamp, 'default': 0},
            "enddate": {'type': self.timestamp, 'default': 0},
            "productcode": {'type': self.product_encoding, 'default': 0},
            "pgamount": {'type': float, 'default': 0},
            "chargetypeid": {'type': float, 'default': 0},
            "concurrentwatchcount": {'type': float, 'default': 0},
            "promo_100": {'type': self.alphabet_encoding, 'default': 0},
            "coinReceived": {'type': self.alphabet_encoding, 'default': 0},
            "Repurchase": {'type': float, 'default': 0},
            "devicetypeid": {'type': self.device_encoding, 'default': 0},
            "isauth": {'type':  self.alphabet_encoding, 'default': 0},
            "gender": {'type':  self.alphabet_encoding, 'default': 0},
            "agegroup": {'type': float, 'default': 0},
            "('channeltype', 'count')": {'type': float, 'default': 0},
            "('channeltype', 'nunique')": {'type': float, 'default': 0},
            "('channeltype', 'mode')": {'type': self.alphabet_encoding, 'default': 0},
            "('devicetype', 'count')": {'type': float, 'default': 0},
            "('devicetype', 'nunique')": {'type': float, 'default': 0},
            "('devicetype', 'mode')": {'type': float, 'default': 0},
            "('hour', 'sum')_x": {'type': None, 'default': 0},
            "('hour', 'mean')_x": {'type': None, 'default': 0},
            "('hour', 'median')_x": {'type': None, 'default': 0},
            "('hour', 'min')_x": {'type': None, 'default': 0},
            "('hour', 'max')_x": {'type': None, 'default': 0},
            "('hour', 'mode')_x": {'type': None, 'default': 0},
            "('hour', 'std')_x": {'type': None, 'default': 0},
            "('hour', 'q25')_x": {'type': None, 'default': 0},
            "('hour', 'q75')_x": {'type': None, 'default': 0},
            "('dates_sec', 'sum')": {'type': None, 'default': 0},
            "('dates_sec', 'mean')": {'type': float, 'default': 0},
            "('dates_sec', 'median')": {'type': float, 'default': 0},
            "('dates_sec', 'min')": {'type': float, 'default': 0},
            "('dates_sec', 'max')": {'type': float, 'default': 0},
            "('dates_sec', 'mode')": {'type': float, 'default': 0},
            "('dates_sec', 'std')": {'type': float, 'default': 0},
            "('dates_sec', 'q25')": {'type': float, 'default': 0},
            "('dates_sec', 'q75')": {'type': float, 'default': 0},
            "('viewtime', 'sum')_x": {'type': float, 'default': 0},
            "('viewtime', 'mean')_x": {'type': float, 'default': 0},
            "('viewtime', 'median')_x": {'type': float, 'default': 0},
            "('viewtime', 'min')_x": {'type': float, 'default': 0},
            "('viewtime', 'max')_x": {'type': float, 'default': 0},
            "('viewtime', 'mode')_x": {'type': float, 'default': 0},
            "('viewtime', 'std')_x": {'type': float, 'default': 0},
            "('viewtime', 'q25')_x": {'type': float, 'default': 0},
            "('viewtime', 'q75')_x": {'type': float, 'default': 0},
            "('programid', 'count')_x": {'type': float, 'default': 0},
            "('programid', 'nunique')_x": {'type': float, 'default': 0},
            "('programid', 'mode')_x": {'type': None, 'default': 0},
            "('contentid', 'count')_x": {'type': float, 'default': 0},
            "('contentid', 'nunique')_x": {'type': float, 'default': 0},
            "('contentid', 'mode')_x": {'type': None, 'default': 0},
            "('hour', 'sum')_y": {'type': None, 'default': 0},
            "('hour', 'mean')_y": {'type': None, 'default': 0},
            "('hour', 'median')_y": {'type': None, 'default': 0},
            "('hour', 'min')_y": {'type': None, 'default': 0},
            "('hour', 'max')_y": {'type': None, 'default': 0},
            "('hour', 'mode')_y": {'type': None, 'default': 0},
            "('hour', 'std')_y": {'type': None, 'default': 0},
            "('hour', 'q25')_y": {'type': None, 'default': 0},
            "('hour', 'q75')_y": {'type': None, 'default': 0},
            "('viewtime', 'sum')_y": {'type': float, 'default': 0},
            "('viewtime', 'mean')_y": {'type': float, 'default': 0},
            "('viewtime', 'median')_y": {'type': float, 'default': 0},
            "('viewtime', 'min')_y": {'type': float, 'default': 0},
            "('viewtime', 'max')_y": {'type': float, 'default': 0},
            "('viewtime', 'mode')_y": {'type': float, 'default': 0},
            "('viewtime', 'std')_y": {'type': float, 'default': 0},
            "('viewtime', 'q25')_y": {'type': float, 'default': 0},
            "('viewtime', 'q75')_y": {'type': float, 'default': 0},
            "('programid', 'count')_y": {'type': float, 'default': 0},
            "('programid', 'nunique')_y": {'type': float, 'default': 0},
            "('programid', 'mode')_y": {'type': None, 'default': 0},
            "('hour', 'sum')_x_": {'type': None, 'default': 0},
            "('hour', 'mean')_x_": {'type': None, 'default': 0},
            "('hour', 'median')_x_": {'type': None, 'default': 0},
            "('hour', 'min')_x_": {'type': None, 'default': 0},
            "('hour', 'max')_x_": {'type': None, 'default': 0},
            "('hour', 'mode')_x_": {'type': None, 'default': 0},
            "('hour', 'std')_x_": {'type': None, 'default': 0},
            "('hour', 'q25')_x_": {'type': None, 'default': 0},
            "('hour', 'q75')_x_": {'type': None, 'default': 0},
            "('viewtime', 'sum')_x_": {'type': float, 'default': 0},
            "('viewtime', 'mean')_x_": {'type': float, 'default': 0},
            "('viewtime', 'median')_x_": {'type': float, 'default': 0},
            "('viewtime', 'min')_x_": {'type': float, 'default': 0},
            "('viewtime', 'max')_x_": {'type': float, 'default': 0},
            "('viewtime', 'mode')_x_": {'type': float, 'default': 0},
            "('viewtime', 'std')_x_": {'type': float, 'default': 0},
            "('viewtime', 'q25')_x_": {'type': float, 'default': 0},
            "('viewtime', 'q75')_x_": {'type': float, 'default': 0},
            "('programid', 'count')": {'type': float, 'default': 0},
            "('programid', 'nunique')": {'type': float, 'default': 0},
            "('programid', 'mode')": {'type': None, 'default': 0},
            "('hour', 'sum')_y_": {'type': None, 'default': 0},
            "('hour', 'mean')_y_": {'type': None, 'default': 0},
            "('hour', 'median')_y_": {'type': None, 'default': 0},
            "('hour', 'min')_y_": {'type': None, 'default': 0},
            "('hour', 'max')_y_": {'type': None, 'default': 0},
            "('hour', 'mode')_y_": {'type': None, 'default': 0},
            "('hour', 'std')_y_": {'type': None, 'default': 0},
            "('hour', 'q25')_y_": {'type': None, 'default': 0},
            "('hour', 'q75')_y_": {'type': None, 'default': 0},
            "('viewtime', 'sum')_y_": {'type': float, 'default': 0},
            "('viewtime', 'mean')_y_": {'type': float, 'default': 0},
            "('viewtime', 'median')_y_": {'type': float, 'default': 0},
            "('viewtime', 'min')_y_": {'type': float, 'default': 0},
            "('viewtime', 'max')_y_": {'type': float, 'default': 0},
            "('viewtime', 'mode')_y_": {'type': float, 'default': 0},
            "('viewtime', 'std')_y_": {'type': float, 'default': 0},
            "('viewtime', 'q25')_y_": {'type': float, 'default': 0},
            "('viewtime', 'q75')_y_": {'type': float, 'default': 0},
            "('total_viewtime', 'sum')": {'type': float, 'default': 0},
            "('total_viewtime', 'mean')": {'type': float, 'default': 0},
            "('total_viewtime', 'median')": {'type': float, 'default': 0},
            "('total_viewtime', 'min')": {'type': float, 'default': 0},
            "('total_viewtime', 'max')": {'type': float, 'default': 0},
            "('total_viewtime', 'mode')": {'type': float, 'default': 0},
            "('total_viewtime', 'std')": {'type': float, 'default': 0},
            "('total_viewtime', 'q25')": {'type': float, 'default': 0},
            "('total_viewtime', 'q75')": {'type': float, 'default': 0},
            "('watch_ratio', 'sum')": {'type': float, 'default': 0},
            "('watch_ratio', 'mean')": {'type': float, 'default': 0},
            "('watch_ratio', 'median')": {'type': float, 'default': 0},
            "('watch_ratio', 'min')": {'type': float, 'default': 0},
            "('watch_ratio', 'max')": {'type': float, 'default': 0},
            "('watch_ratio', 'mode')": {'type': float, 'default': 0},
            "('watch_ratio', 'std')": {'type': float, 'default': 0},
            "('watch_ratio', 'q25')": {'type': float, 'default': 0},
            "('watch_ratio', 'q75')": {'type': float, 'default': 0},
            "('contentid', 'count')_y": {'type': float, 'default': 0},
            "('contentid', 'nunique')_y": {'type': float, 'default': 0},
            "('contentid', 'mode')_y": {'type': None, 'default': 0},
            "('viewtime', 'sum')": {'type': float, 'default': 0},
            "('viewtime', 'mean')": {'type': float, 'default': 0},
            "('viewtime', 'median')": {'type': float, 'default': 0},
            "('viewtime', 'min')": {'type': float, 'default': 0},
            "('viewtime', 'max')": {'type': float, 'default': 0},
            "('viewtime', 'mode')": {'type': float, 'default': 0},
            "('viewtime', 'std')": {'type': float, 'default': 0},
            "('viewtime', 'q25')": {'type': float, 'default': 0},
            "('viewtime', 'q75')": {'type': float, 'default': 0},
            "('pgamount', 'sum')_x": {'type': float, 'default': 0},
            "('pgamount', 'mean')_x": {'type': float, 'default': 0},
            "('pgamount', 'median')_x": {'type': float, 'default': 0},
            "('pgamount', 'min')_x": {'type': float, 'default': 0},
            "('pgamount', 'max')_x": {'type': float, 'default': 0},
            "('pgamount', 'mode')_x": {'type': float, 'default': 0},
            "('pgamount', 'std')_x": {'type': float, 'default': 0},
            "('pgamount', 'q25')_x": {'type': float, 'default': 0},
            "('pgamount', 'q75')_x": {'type': float, 'default': 0},
            "('devicetypeid', 'count')": {'type': float, 'default': 0},
            "('devicetypeid', 'nunique')": {'type': float, 'default': 0},
            "('devicetypeid', 'mode')": {'type': self.device_encoding, 'default': 0},
            "('productcode', 'count')_x": {'type': float, 'default': 0},
            "('productcode', 'nunique')_x": {'type': float, 'default': 0},
            "('productcode', 'mode')_x": {'type': self.product_encoding, 'default': 0},
            "('productcode', 'count')_y": {'type': float, 'default': 0},
            "('productcode', 'nunique')_y": {'type': float, 'default': 0},
            "('productcode', 'mode')_y": {'type': self.product_encoding, 'default': 0},
            "('totalamount', 'sum')": {'type': float, 'default': 0},
            "('totalamount', 'mean')": {'type': float, 'default': 0},
            "('totalamount', 'median')": {'type': float, 'default': 0},
            "('totalamount', 'min')": {'type': float, 'default': 0},
            "('totalamount', 'max')": {'type': float, 'default': 0},
            "('totalamount', 'mode')": {'type': float, 'default': 0},
            "('totalamount', 'std')": {'type': float, 'default': 0},
            "('totalamount', 'q25')": {'type': float, 'default': 0},
            "('totalamount', 'q75')": {'type': float, 'default': 0},
            "('pgamount', 'sum')_y": {'type': float, 'default': 0},
            "('pgamount', 'mean')_y": {'type': float, 'default': 0},
            "('pgamount', 'median')_y": {'type': float, 'default': 0},
            "('pgamount', 'min')_y": {'type': float, 'default': 0},
            "('pgamount', 'max')_y": {'type': float, 'default': 0},
            "('pgamount', 'mode')_y": {'type': float, 'default': 0},
            "('pgamount', 'std')_y": {'type': float, 'default': 0},
            "('pgamount', 'q25')_y": {'type': float, 'default': 0},
            "('pgamount', 'q75')_y": {'type': float, 'default': 0},
            "('coinamount', 'sum')": {'type': float, 'default': 0},
            "('coinamount', 'mean')": {'type': float, 'default': 0},
            "('coinamount', 'median')": {'type': float, 'default': 0},
            "('coinamount', 'min')": {'type': float, 'default': 0},
            "('coinamount', 'max')": {'type': float, 'default': 0},
            "('coinamount', 'mode')": {'type': float, 'default': 0},
            "('coinamount', 'std')": {'type': float, 'default': 0},
            "('coinamount', 'q25')": {'type': float, 'default': 0},
            "('coinamount', 'q75')": {'type': float, 'default': 0},
            "('bonusamount', 'sum')": {'type': float, 'default': 0},
            "('bonusamount', 'mean')": {'type': float, 'default': 0},
            "('bonusamount', 'median')": {'type': float, 'default': 0},
            "('bonusamount', 'min')": {'type': float, 'default': 0},
            "('bonusamount', 'max')": {'type': float, 'default': 0},
            "('bonusamount', 'mode')": {'type': float, 'default': 0},
            "('bonusamount', 'std')": {'type': float, 'default': 0},
            "('bonusamount', 'q25')": {'type': float, 'default': 0},
            "('bonusamount', 'q75')": {'type': float, 'default': 0},
            "('discountamount', 'sum')": {'type': float, 'default': 0},
            "('discountamount', 'mean')": {'type': float, 'default': 0},
            "('discountamount', 'median')": {'type': float, 'default': 0},
            "('discountamount', 'min')": {'type': float, 'default': 0},
            "('discountamount', 'max')": {'type': float, 'default': 0},
            "('discountamount', 'mode')": {'type': float, 'default': 0},
            "('discountamount', 'std')": {'type': float, 'default': 0},
            "('discountamount', 'q25')": {'type': float, 'default': 0},
            "('discountamount', 'q75')": {'type': float, 'default': 0},
        }
        return [item['type'] for item in encodings.values()]


def main(filename):
    encoding: list[Callable] = Encoding().get_encodings()
    with open(f'./data/{filename}.csv', 'r') as file:
        csv_file = StringIO(file.read())
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        if not os.path.isdir('./result'):
            os.makedirs('./result')

        with open(f'./result/{filename}_result.csv', 'w', newline='') as result_file:
            csv_writer = csv.writer(result_file, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i, row in tqdm(enumerate(csv_reader)):
                newline = []
                for j, col in enumerate(row[1:]):
                    if j > (len(encoding) - 1) or (not encoding[j]):
                        continue
                    if i == 0:
                        newline.append(col)
                    else:
                        try:
                            newline.append(encoding[j](col))
                        except ValueError as e:
                            newline.append(0)
                csv_writer.writerow(newline)


if __name__ == '__main__':
    main('train')
    main('test')
