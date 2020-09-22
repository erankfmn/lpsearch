"""
This examples shows our fractional lp search compared to other methods.
we choose randomly different points and compare to other methods
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from LpSearch import *


def distance_matrix(x,y,order):
    length = len(x)
    matrix = np.empty([length,length])
    for i in range(length):
        for j in range(length):
            matrix[i,j]=np.sum(np.abs(x[i]-y[j])**order)
    return matrix





# Initialize size of the database, iterations and required neighbors.
n_samples = 10000
n_features = 10
n_queries = 3
n_hashtables=10
index_size=5
n_neighbors=1

# Generate sample data
X, _ = make_blobs(n_samples=n_samples + n_queries,
                  n_features=n_features, centers=10,
                  random_state=0)

X_index = X[:n_samples]
X_query = X[n_samples:]


# nbrs2 = NearestNeighbors(n_neighbors=1, algorithm='kd_tree',
#                         metric="euclidean").fit(y_index)
#
# neighbors_exact2 = nbrs2.kneighbors(y_query, return_distance=False)

P = [0.5,0.6,0.7,0.8,0.9]
results1=[]
results2=[]

for p in P :

    # Get exact neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute',
                            metric=lambda x,y : np.sum(np.abs(x-y)**p)**(1/p)).fit(X_index)



    exact_sum = 0

    neighbors_exact = nbrs.kneighbors(X_query, return_distance=True)
    for i in range(n_neighbors):
        exact_sum += neighbors_exact[0][0][i]

    search_lazy = LpSearch(p,index_size,num_hashtables=n_hashtables,lazy=True)
    search_lazy.index(X_index)

    lazy_sum=0

    a =search_lazy.query(X_query[0],n_neighbors)
    for i in range(n_neighbors):
        lazy_sum+= a[i][1]

    search_frac = LpSearch(p,index_size,num_hashtables=n_hashtables)
    search_frac.index(X_index)

    frac_sum = 0

    b = search_frac.query(X_query[0], n_neighbors)
    for i in range(n_neighbors):
        frac_sum += b[i][1]



 #   print(neighbors_exact,a,b)
    results1.append(lazy_sum - exact_sum)
    results2.append(frac_sum - exact_sum)


plt.plot(P, results1,'*-',label="lazy lsh")
plt.plot(P, results2,'*-',label="our lsh")
#plt.xticks(a)
plt.title('difference in distance')
plt.legend()

#lshf = LSHash(2*math.ceil(math.log(dim,2)), dim)
#lshf = lsh = MinHashLSH(threshold=0.5, num_perm=dim)

# m2 = MinHash(dim)
#
# for i,item in enumerate(y_index):
#     for corr in item:
#         m2.update(corr)
#     lshf.insert(i,m2)
#
# approx=[]
# for item in y_query:
#     approx.append(lshf.query(item))
#
#
#
# # Set `n_candidate` values
# n_candidates_values = np.linspace(10, 500, 5).astype(np.int)
# n_estimators_for_candidate_value = [1, 5, 10]
# n_iter = 10
# stds_accuracies = np.zeros((len(n_estimators_for_candidate_value),
#                             n_candidates_values.shape[0]),
#                            dtype=float)
# accuracies_c = np.zeros((len(n_estimators_for_candidate_value),
#                          n_candidates_values.shape[0]), dtype=float)
#
# # LSH Forest is a stochastic index: perform several iteration to estimate
# # expected accuracy and standard deviation displayed as error bars in
# # the plots
# for j, value in enumerate(n_estimators_for_candidate_value):
#     for i, n_candidates in enumerate(n_candidates_values):
#         accuracy_c = []
#         for seed in range(n_iter):
#             lshf = LSHForest(n_estimators=value,
#                              n_candidates=n_candidates, n_neighbors=1,
#                              random_state=seed)
#
#             # Build the LSH Forest index
#             lshf.fit(X_index)
#             # Get neighbors
#             neighbors_approx = lshf.kneighbors(X_query,
#                                                return_distance=False)
#             accuracy_c.append(np.sum(np.equal(neighbors_approx,
#                                               neighbors_exact)) /
#                               n_queries)
#
#         stds_accuracies[j, i] = np.std(accuracy_c)
#         accuracies_c[j, i] = np.mean(accuracy_c)
#
# # Set `n_estimators` values
# n_estimators_values = [1, 5, 10, 20, 30, 40, 50]
# accuracies_trees = np.zeros(len(n_estimators_values), dtype=float)
#
# # Calculate average accuracy for each value of `n_estimators`
# for i, n_estimators in enumerate(n_estimators_values):
#     lshf = LSHForest(n_estimators=n_estimators, n_neighbors=1)
#     # Build the LSH Forest index
#     lshf.fit(X_index)
#     # Get neighbors
#     neighbors_approx = lshf.kneighbors(X_query, return_distance=False)
#     accuracies_trees[i] = np.sum(np.equal(neighbors_approx,
#                                           neighbors_exact))/n_queries
#
# ###############################################################################
# # Plot the accuracy variation with `n_candidates`
# plt.figure()
# colors = ['c', 'm', 'y']
# for i, n_estimators in enumerate(n_estimators_for_candidate_value):
#     label = 'n_estimators = %d ' % n_estimators
#     plt.plot(n_candidates_values, accuracies_c[i, :],
#              'o-', c=colors[i], label=label)
#     plt.errorbar(n_candidates_values, accuracies_c[i, :],
#                  stds_accuracies[i, :], c=colors[i])
#
# plt.legend(loc='upper left', fontsize='small')
# plt.ylim([0, 1.2])
# plt.xlim(min(n_candidates_values), max(n_candidates_values))
# plt.ylabel("Accuracy")
# plt.xlabel("n_candidates")
# plt.grid(which='both')
# plt.title("Accuracy variation with n_candidates")
#
# # Plot the accuracy variation with `n_estimators`
# plt.figure()
# plt.scatter(n_estimators_values, accuracies_trees, c='k')
# plt.plot(n_estimators_values, accuracies_trees, c='g')
# plt.ylim([0, 1.2])
# plt.xlim(min(n_estimators_values), max(n_estimators_values))
# plt.ylabel("Accuracy")
# plt.xlabel("n_estimators")
# plt.grid(which='both')
# plt.title("Accuracy variation with n_estimators")
#
plt.show()

