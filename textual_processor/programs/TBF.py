from __future__ import division, unicode_literals
import hashlib


class TBF(object):

    """Textual Bloom Filter"""

    def __init__(self,l=1000, k=20):
        self.l = l # length of bloom filter
        self.k = k # number of hash functions

        self.cbf_tf = [0] * l
        self.cbf_idf = [0] * l

        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5

    # add list of tokens (textual data) and corresponding token frequencies
    def add_list(self, term_list, tf_list, idf_list):

        for val in term_list:
            hex_str1 = self.h1(val).hexdigest()
            int1 = int(hex_str1, 16)
            hex_str2 = self.h2(val).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(self.k):
                gi = int1 + i * int2
                gi = int(gi % self.l)
                # create counting bloom filter with tf values
                self.cbf_tf[gi] += tf_list[term_list.index(val)]
                # create counting bloom filter with idf values
                self.cbf_idf[gi] += idf_list[term_list.index(val)]

        return self.cbf_tf, self.cbf_idf

    # Add a single token to the counting bloom filter
    def add_item(self, item):
        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)
            self.cbf_tf[gi] += 1

    # query an item's existence and frequency
    def query_item(self, item):

        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        freqs = []

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)

            if self.cbf_tf[gi] >= 1:
                freqs.append(self.cbf_tf[gi])
            else:
                return False, 0

        return True, min(freqs)

    # remove an item from cbf_tf
    def remove_item(self, item):

        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)
            self.cbf_tf[gi] -= 1

    # Calculate (Dice's coefficient) similarity  between a counting bloom filter and a list of tokens.
    def cal_dc_sim_list(self, len_list1, list2):

        comm_tokens = 0
        for s in list2:
            status, freq = self.query_item(s)
            if status == True:
                comm_tokens += 1

        # Dice's Coefficient
        sim = 2 * comm_tokens / int(len_list1) + len(list2)
        return sim

    # Calculate (Jaccard's Index) similarity between a counting bloom filter and a list of tokens
    def cal_ji_sim_list(self, len_list1, list2):

        comm_tokens = 0
        for s in list2:
            status, freq = self.query_item(s)
            if status == True:
                comm_tokens += 1

        # Dice's Coefficient
        sim = comm_tokens / (int(len_list1) + len(list2) - comm_tokens)
        return sim

    # calculate dissimilarity between two counting bloom filters (between two textual data)
    def cal_dissim_cbf_tf(self, cbf2):

        assert len(self.cbf_tf) != len(cbf2), 'bloom filters are not of same length'

        if len(self.cbf_tf) == len(cbf2):
            sum_diff = 0
            for i in range(self.k):
                sum_diff += abs(self.cbf_tf[i] - cbf2[i])

            # calculation of dissimilarity
            dissim = sum_diff / (2 * self.k)
        else:
            return None

        return dissim

    # calculate dice coefficient similarity between  two counting bloom filters
    def cal_dc_sim_cbf(self, cbf2):

        assert len(self.cbf_tf) != len(cbf2), 'bloom filters are not of same length'

        if len(self.cbf_tf) == len(cbf2):
            sum_min = 0
            for x,y in zip(self.cbf_tf, cbf2):
                sum_min += min(x,y)
        else:
            return None

        return 2 * sum_min / sum(self.cbf_tf) + sum(cbf2)








