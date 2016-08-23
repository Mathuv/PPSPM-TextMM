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

    sum_min = sum([min(i, j) for i, j in zip(cbf1, cbf2)])

    return 2 * sum_min / (sum(cbf1) + sum(cbf2))


class TBF(object):

    """Textual Bloom Filter"""

    def __init__(self, l=1000, k=20):
        self.l = l  # length of bloom filter
        self.k = k  # number of hash functions

        self.cbf_freq = [0] * l
        self.cbf_idf = [0] * l

        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
  
    def add_list_tfidf(self, term_list, freq_list, idf_list=None):
        """add list of tokens (textual data) and corresponding token frequencies
        and their idf (inverse document frequencies)"""

        for idx, val in enumerate(term_list):
            hex_str1 = self.h1(val).hexdigest()
            int1 = int(hex_str1, 16)
            hex_str2 = self.h2(val).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(self.k):
                gi = int1 + i * int2
                gi = int(gi % self.l)
                # create counting bloom filter with tf values
                self.cbf_freq[gi] += freq_list[idx]
                # create counting bloom filter with idf values
                if idf_list:
                    self.cbf_idf[gi] += idf_list[idx]

        return (self.cbf_freq, self.cbf_idf) if idf_list else self.cbf_freq
