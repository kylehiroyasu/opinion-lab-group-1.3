import unittest
import torch as t

from Model import Model

class TestTroughput(unittest.TestCase):

    def setUp(self):
        self.data = t.tensor([[[1.0, 0], [2, 0]],
                              [[0, 1], [0, 5]],
                              [[10, 3], [2, 10]]])
        self.matrix = t.tensor([[1, 0.5],
                                [3, 5]])
        self.model = Model(2, 2)

    def test_avg(self):
        self.average = t.tensor([[1.5, 0],
                                 [0, 3],
                                 [6, 6.5]])
        avg = self.model.average(self.data)
        self.assertEqual(self.average.size(), avg.size())
        for i in range(self.average.size()[0]):
            for j in range(self.average.size()[1]): 
                self.assertEqual(self.average[i,j], avg[i,j])

    def test_logit(self):
        self.logit = t.tensor([[1.5, 4.5],
                               [1.5, 15],
                               [9.25, 50.5]])
        avg = self.model.average(self.data)
        logit = self.model.attention_logit(avg, self.matrix)
        self.assertEqual(self.logit.size(), logit.size())
        for i in range(self.logit.size()[0]):
            for j in range(self.logit.size()[1]):
                self.assertEqual(self.logit[i,j], logit[i,j])

    def test_d(self):
        self.d = t.tensor([[1.5, 3],
                            [15, 75],
                            [244, 523.5]])
        avg = self.model.average(self.data)
        logit = self.model.attention_logit(avg, self.matrix)
        d = self.model.attention_d(self.data, logit)
        self.assertEqual(self.d.size(), d.size())
        for i in range(self.d.size()[0]):
            for j in range(self.d.size()[1]):
                self.assertEqual(self.d[i,j], d[i,j])

    def test_weighted_sum(self):
        self.weights = t.tensor([[0.5, 0.5],
                                  [0, 1],
                                  [0.25, 0.75]])
        self.result = t.tensor([[1.5, 0],
                                [0, 6],
                                [3, 9.75]])
        result = self.model.weighted_sum(self.weights, self.data)
        self.assertEqual(self.result.size(), result.size())
        for i in range(self.result.size()[0]):
            for j in range(self.result.size()[1]):
                self.assertEqual(self.result[i,j], result[i,j], ""+str(i)+", "+str(j))

if __name__ == '__main__':
    unittest.main()