from __future__ import division, unicode_literals
import hashlib


def mcalc_sim_tf_idf(cbf_tf1, cbf_idf1, cbf_tf2, cbf_idf2):
    """Calculate DC similarity with both tf (term frequency) and idf (inverse document frequency)"""
    sum_min = 0
    div_cbf1 = 0
    div_cbf2 = 0

    for q1, q2, d1, d2 in zip(cbf_tf1, cbf_idf1, cbf_tf2, cbf_idf2):
        sum_min += min(q1, d1) * ((q2 + d2) / 2)
        div_cbf1 += q1 * q2
        div_cbf2 += d1 * d2

    if div_cbf1 + div_cbf2:
        return 2 * sum_min / (div_cbf1 + div_cbf2)
    else:
        return 0.0


def mcalc_sim_freq(cbf1, cbf2):
    """Calculate DC similarity only with tf (term frequency)"""
    sum_min = 0

    # for q, d in zip(cbf1, cbf2):
    #     sum_min += min(q, d)

    sum_min = sum([min(i, j) for i, j in zip(cbf1, cbf2)])

    return 2 * sum_min / (sum(cbf1) + sum(cbf2))



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










