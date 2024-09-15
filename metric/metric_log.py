from typing import List, Dict

import numpy as np

class MetricLog: 
    metric_data: Dict[str, List]
    def __init__(self):
        self.metric_data = dict()

    def add_metrics(self, metrics: Dict):
        for key, val in metrics.items():
            if key not in self.metric_data:
                self.__add_metric_category(key)

            self.metric_data[key].append(val)

    def __add_metric_category(self, key: str):
        self.metric_data[key] = []

    def get_metric(self, key: str):
        metric = np.mean(np.stack(self.metric_data[key], axis=0))

        return metric
 
    @property
    def all_metric_categories(self):
        metric_categories = list(self.metric_data.keys())

        return metric_categories