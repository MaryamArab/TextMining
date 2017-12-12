import matplotlib.pyplot as plt
import pandas as pd
import pyclust
import numpy

from sklearn.cluster import KMeans

vectors = pd.read_csv("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/My-Word2Vec/Ipynb/data" )
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform( vectors )

print("Compute structured Bisecting-Kmeans clustering...")

#cluster_range = range( 2, 6 )
cluster_range2 = range( 2, 65)

cluster_errors = []
data= []
for num_clusters in cluster_range2:
    #print ("?????????", num_clusters)
    clusters = pyclust.BisectKMeans(num_clusters)
    clusters.fit(vectors.iloc[:, :].values)
    #clusters.fit_predict(vectors)

    #clusters = pyclust.Bisect(num_clusters).fit(vectors)
    #cluster_errors.append( clusters.inertia_ )
    #cluster_errors.append(clusters.sse_arr_)
    #print ("SSE",clusters.sse_arr_)
    data.append(sum(clusters.sse_arr_.values()))
    print ("ssse",data)


seri = clusters.sse_arr_
print ("Total Clustering SSE", data)
#print ("!!!!!!!!!!", clusters.sse_arr_)

#print ("Seriiiiiii", seri.values())
clusters_df = pd.DataFrame.from_dict( {"num_clusters":cluster_range2, "cluster_errors": data})


print (clusters_df[0:10])

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.show()
print ("***done****")