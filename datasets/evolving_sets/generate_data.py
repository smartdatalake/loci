from random import randint, choices, seed

seed(1924)

no_snapshots = 10
no_members = 100
no_clusters_min = 10
no_clusters_max = 20


with open('./data/data.csv', 'w') as o:
    for i in range(no_snapshots):
        no_clusters = randint(no_clusters_min, no_clusters_max)
        clustering = choices(range(no_clusters), k=no_members)
        
        for no, mem in enumerate(clustering):
            o.write("{},{},{}\n".format(i, mem, no))