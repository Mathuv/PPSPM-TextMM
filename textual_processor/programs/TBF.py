from __future__ import division, unicode_literals
import hashlib


class TBF(object):

    """Textual Bloom Filter"""

    def __init__(self,l=1000, k=20):
        self.l = l # length of bloom filter
        self.k = k # number of hash functions

        self.cbf_freq = [0] * l
        self.cbf_idf = [0] * l

        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
        
    # DV: I think only the function add_list_tfidf is sufficient, not the add_list and add_item functions!
    # when freq_list and idf_list are set to None, you can generate CBFs as with add_list function

    def add_list_tfidf(self, term_list, freq_list, idf_list):
        """add list of tokens (textual data) and corresponding token frequencies
        and their idf (inverse document frequencies)"""

        for val in term_list:
            hex_str1 = self.h1(val).hexdigest()
            int1 = int(hex_str1, 16)
            hex_str2 = self.h2(val).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(self.k):
                gi = int1 + i * int2
                gi = int(gi % self.l)
                # create counting bloom filter with tf values
                self.cbf_freq[gi] += freq_list[term_list.index(val)]
                # create counting bloom filter with idf values
                self.cbf_idf[gi] += idf_list[term_list.index(val)]

        return self.cbf_freq, self.cbf_idf

    def add_list(self, term_list, freq_list):
        """add list of tokens (textual data) and corresponding token frequencies"""

        for val in term_list:
            hex_str1 = self.h1(val).hexdigest()
            int1 = int(hex_str1, 16)
            hex_str2 = self.h2(val).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(self.k):
                gi = int1 + i * int2
                gi = int(gi % self.l)
                self.cbf_freq[gi] += freq_list[term_list.index(val)]

        return self.cbf_freq

    def add_item(self, item):
        """Add a single token to the counting bloom filter"""
        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)
            self.cbf_freq[gi] += 1

    def query_item(self, item):
        """query an item's existence and frequency"""

        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        freqs = []

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)

            if self.cbf_freq[gi] >= 1:
                freqs.append(self.cbf_freq[gi])
            else:
                return False, 0

        return True, min(freqs)

    def remove_item(self, item):
        """remove an item from cbf_freq"""
        """DV: why this function is required?"""

        hex_str1 = self.h1(item).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(item).hexdigest()
        int2 = int(hex_str2, 16)

        for i in range(self.k):
            gi = int1 + i * int2
            gi = int(gi % self.l)
            self.cbf_freq[gi] -= 1

    def cal_dc_sim_list(self, len_list1, list2):
        """Calculate (Dice's coefficient) similarity  between a counting bloom filter and a list of tokens."""

        comm_tokens = 0
        for s in list2:
            status, freq = self.query_item(s)
            if status == True:
                comm_tokens += 1

        # Dice's Coefficient
        sim = 2 * comm_tokens / int(len_list1) + len(list2)
        return sim
        
        #DV: this is correct, but now with TF, and TF_IDF weightings the functions need to be changed, right?
        #DV: maybe move functions like mcalc_sim_tf_idf from textprocessor.py to here

    def cal_ji_sim_list(self, len_list1, list2):
        """Calculate (Jaccard's Index) similarity between a counting bloom filter and a list of tokens"""

        comm_tokens = 0
        for s in list2:
            status, freq = self.query_item(s)
            if status == True:
                comm_tokens += 1

        # Dice's Coefficient
        sim = comm_tokens / (int(len_list1) + len(list2) - comm_tokens)
        return sim

    def cal_dissim_cbf_tf(self, cbf2):
        """calculate dissimilarity between two counting bloom filters (between two textual data)"""

        assert len(self.cbf_freq) != len(cbf2), 'bloom filters are not of same length'

        if len(self.cbf_freq) == len(cbf2):
            sum_diff = 0
            for i in range(self.k):
                sum_diff += abs(self.cbf_freq[i] - cbf2[i])

            # calculation of dissimilarity
            dissim = sum_diff / (2 * self.k)
        else:
            return None

        return dissim

    def cal_dc_sim_cbf(self, cbf2):
        """calculate dice coefficient similarity between  two counting bloom filters"""

        assert len(self.cbf_freq) != len(cbf2), 'bloom filters are not of same length'

        if len(self.cbf_freq) == len(cbf2):
            sum_min = 0
            for x,y in zip(self.cbf_freq, cbf2):
                sum_min += min(x,y)
        else:
            return None

        return 2 * sum_min / sum(self.cbf_freq) + sum(cbf2)








