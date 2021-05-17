import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


class AnomalyDetection:
    def __init__(self, ci):
        self.confidence = ci
        self.z_score_thresh = 3
        self.timestamp_col = 'timestamp'
        self.col_to_test = 'packets'
        self.port = 53

        self.min_time_b = None
        self.max_time_b = None
        self.min_time_s = None
        self.max_time_s = None
        self.max_spike = None
        self.ts_of_last_spike = None
        return

    def find_indexes_of_spikes(self, data):
        indexes_of_spikes = (data.z > self.z_score_thresh)
        # indexes_of_spikes.iloc[0] = True
        time_intervals = (np.diff(data.loc[indexes_of_spikes, self.timestamp_col])).reshape(-1, 1)

        # spikes have small and large gaps so will cluster them to 2 groups using KMEANS
        time_clusters = KMeans(n_clusters=2, random_state=0).fit(time_intervals).labels_
        print("mean time interval: ", np.mean(time_intervals[time_clusters == 0]),
              np.mean(time_intervals[time_clusters == 1]))
        print("std time interval: ", np.std(time_intervals[time_clusters == 0]),
              np.std(time_intervals[time_clusters == 1]))

        # choose the one with the larger mean:
        choose_0 = np.mean(time_intervals[time_clusters == 0]) > np.mean(time_intervals[time_clusters == 1])
        if choose_0:
            big_time_interval = time_intervals[time_clusters == 0]
            small_time_interval = time_intervals[time_clusters == 1]
        else:
            big_time_interval = time_intervals[time_clusters == 1]
            small_time_interval = time_intervals[time_clusters == 0]

        return indexes_of_spikes, big_time_interval, small_time_interval

    def mean_confidence_interval(self, data):
        # calculate boundries of time range and spike values with confidence interval
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + self.confidence) / 2., n - 1)
        return m - h, m + h

    def preprocess_data(self, train_df):
        # calculate normal ranges for time gaps and spike height
        z = np.abs(stats.zscore(train_df[self.col_to_test].values.reshape(-1)))
        train_df['z'] = z

        indexes_of_spikes, big_time_interval, small_time_interval = self.find_indexes_of_spikes(train_df)

        self.min_time_b, self.max_time_b = self.mean_confidence_interval(big_time_interval)
        _, self.max_time_s = self.mean_confidence_interval(small_time_interval)

        _, self.max_spike = self.mean_confidence_interval(train_df.loc[indexes_of_spikes, self.col_to_test].values)
        print(self.max_spike)

        # find last non anomalous spike
        self.ts_of_last_spike = train_df.loc[indexes_of_spikes[indexes_of_spikes].index[-1]].timestamp
        return

    def check_if_anomaly(self, new_data):
        # return True if anomaly, else False

        # check if numer of packets is consistent:
        if new_data.packets != new_data.packets_sent + new_data.packets_received:
            print("numer of packets is not consistent")
            return True
        # check if there is a spike:
        if new_data.packets > self.max_spike:
            # check if spike outside learned range:
            if new_data.timestamp - self.ts_of_last_spike > self.max_time_s:
                if new_data.timestamp - self.ts_of_last_spike > self.max_time_b or new_data.timestamp - self.ts_of_last_spike < self.min_time_b:
                    print("spike outside learned range")
                    return True
                else:
                    # new normal spike so update ts_of_last_spike
                    print("update ts_of_last_spike")
                    self.ts_of_last_spike = new_data.timestamp
            else:
                # spike within allowed range so update ts_of_last_spike
                print("update ts_of_last_spike")
                self.ts_of_last_spike = new_data.timestamp

        return False

    def run(self, df, split_ratio):
        df_dns = df.loc[df.dest_port == self.port].copy().reset_index()

        # split data to train and test by keeping last split_ratio% of rows
        partition = int(0.01 * split_ratio * df_dns.shape[0])
        train_df = df_dns.iloc[:partition, :].copy()
        test_df = df_dns.iloc[partition:, :].copy()

        self.preprocess_data(train_df)
        anomaly_indexes = []
        for idx in range(test_df.shape[0]):
            if self.check_if_anomaly(test_df.iloc[idx, :]):
                anomaly_indexes.append(test_df.iloc[idx, 0])

        return anomaly_indexes
