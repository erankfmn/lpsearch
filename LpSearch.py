import math
import numpy as np
from bitarray import bitarray
from scipy.stats import cauchy
import pickle
import struct

from datasketch.storage import (
    ordered_storage, unordered_storage, _random_name)



class LpSearch(object):


    def __init__(self, p , hash_size ,epsilon=0.1, num_hashtables=1,lazy=False,
                 storage_config=None):

        self.hash_size = hash_size
        self.num_hashtables = num_hashtables
        self.p =p
        self.epsilon = epsilon
        self.k= 10#1/epsilon #math.ceil(math.log(1 / epsilon, 1 + epsilon))
        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config
        self.lazy=lazy
        self._init_hashtables()

    def mydist(self,x, y):
        return np.sum(np.abs(x - y) ** self.p) ** (1 / self.p)

    def g(self,x, r):
        return np.array([1 - np.cos(x / r), np.sin(x / r)]) / (r ** (-self.p / 2))

    def g_prime(self,x, i):
        sum = [0, 0]
        for j in range( self.k):
            sum += self.g(x, (1 + self.epsilon) ** (i + (j -  self.k/2) * self.k))
        return sum

    def embedd(self,x):
        answer = np.empty([2 * x.shape[0] * self.k])
        for j in range(x.shape[0]):
            list = []
            for l in range(self.k):
                list.extend(self.g_prime(x[j], l))
            answer[j * 2 * self.k:(j + 1) * 2 * self.k] = list
        return answer

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hashtables = [unordered_storage({'type': 'dict'}) for _ in
                      range(self.num_hashtables)]

    def _generate_uniform_planes(self,size,input_dim):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """
        if(self.lazy):
            self.uniform_planes = cauchy.rvs(size=[size, self.hash_size, self.points.shape[1]])
        else:
            self.uniform_planes = np.random.randn(size,self.hash_size,input_dim)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """

        try:
            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSHash instance""", e)
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])

    def index(self, input_points, extra_data=None):
        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table. `extra_data` needs to be JSON serializable if in-memory
        dict is not used as storage.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
            This object will be converted to Python tuple and stored in the
            selected storage.
        :param extra_data:
            (optional) Needs to be a JSON-serializable object: list, dicts and
            basic types such as strings and integers.
        """
        self.points=input_points

        if not hasattr(self, 'uniform_planes'):
            self._generate_uniform_planes(input_points.shape[0],2 * self.k*input_points.shape[1])



        for index in range(input_points.shape[0]):
         for i, table in enumerate(self.hashtables):
             if(self.lazy):
                 table.insert(self._hash(self.uniform_planes[i],input_points[index]), index)
             else :
                 table.insert(self._hash(self.uniform_planes[i], self.embedd(input_points[index])), index)

    def query(self, query_point, num_results=1, distance_func=None):
        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """
        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hashtables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = LpSearch.hamming_dist(key, binary_hash)
                    if distance < 2:
                        candidates.update(table.get_list(key))

            d_func = LpSearch.euclidean_dist_square

        else:

            if distance_func == "euclidean":
                d_func = LpSearch.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LpSearch.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LpSearch.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LpSearch.cosine_dist
            elif distance_func == "l1norm":
                d_func = LpSearch.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")


            for i, table in enumerate(self.hashtables):
                if (self.lazy):
                    binary_hash = self._hash(self.uniform_planes[i], query_point)
                else:
                     binary_hash = self._hash(self.uniform_planes[i], self.embedd(query_point))

                candidates.update(table.get(binary_hash))

            # rank candidates by distance function
            candidates = [[ix, self.mydist(query_point, self.points[ix])]
                          for ix in candidates]
            candidates = sorted(candidates, key=lambda x: x[1])

        return candidates[:num_results]

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

