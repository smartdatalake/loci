import networkx as nx
from collections import Counter
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import json
import ipycytoscape
import random

class Change_Detector:
    """
    Change Detector class for studying evolving sets.

    :param similarities: map for caching group similarities, in the form of (group1, group2) -> sim
    :type similarities: dict
    :param groups: 3d map, in the form of snapshot->group->member
    :type groups: dict
    :param matchings: map for caching snapshot similarities, in the form of (snapshot1, snapshot2) -> (group1, group2, sim)
    :type matchings: dict
    :param inv_index: 3d map, in the form of member->snapshot->group
    :type inv_index: dict
    """
    
    def __init__(self):
        pass
            
    def set_data(self, data, type, file=False):
        """
        Pass data to initialize Change Detector.
        
        :param data: data given or filename to find data.
        :type data: str
        :param type: type of data given, "json" or "csv".
        :type type: str
        :param file: whether file is given for input. Filename is stored in data, defaults to False
        :type file: bool, optional
        :return: None
        :rtype: None

        """
        self.similarities = {}
        self.groups = {}
        self.matchings = {}
        self.inv_index = {}
        
        if type=='json':
            if file==True:
                with open(data) as f:
                    self.groups = json.load(f)
            else:
                self.groups = data
            self.__construct_groups_from_json()
        elif type == 'csv':
            if file==True:
                df = pd.read_csv(data, header=None)
            else:
                df = data  
            df = df.astype(str)                
            self.__construct_groups_from_csv(df)            

            
    def __construct_groups_from_json(self):
        """
        Inner function that transforms a json into the appropriate class structure.
        
        :return: None
        :rtype: None

        """
        for snap in self.groups:
            for group in self.groups[snap]:
                self.groups[snap][group] = set(self.groups[snap][group])
                
                for member in self.groups[snap][group]:
                    if member not in self.inv_index:
                        self.inv_index[member] = {}
                    self.inv_index[member][snap] = group                        

    def __construct_groups_from_csv(self, df):
        """
        Inner function that transforms a pandas dataframe into the appropriate class structure.
        
        :param df: DataFrame into the form of snap,group,member.
        :type df: DataFrame
        :return: None
        :rtype: None

        """
        df = df.groupby([0, 1])[2].apply(set)

        self.groups = {}
        for (snap, group), row in df.iteritems():
            if snap not in self.groups:
                self.groups[snap] = {}
            self.groups[snap][group] = row
            
            for member in self.groups[snap][group]:
                if member not in self.inv_index:
                    self.inv_index[member] = {}
                self.inv_index[member][snap] = group   


    def __sum_matching(self, snap1, snap2):
        """
        Inner function that calculates sum of matching.
        
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :return: Sum of matching of two snapshots.
        :rtype: int

        """
        snap_key = (snap1, snap2)
        matching = 0
        for group1, group2, sim in self.matchings[snap_key]:
            matching += sim
            
        score = matching / (len(self.groups[snap1]) + len(self.groups[snap2]) - matching)
        return score    
    
    def __calculate_all_group_similarities(self, snap1, snap2):
        """
        Inner function that calculates all jaccards between groups.
        
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :return: All jacards of all groups of two snapshots.
        :rtype: list

        """
        key = (snap1, snap2)
        if key in self.similarities:
            return self.similarities[key]
        
        self.similarities[key] = {}
        for group1 in self.groups[snap1]:
            for group2 in self.groups[snap2]:
                if (group1, group2) not in self.similarities[key]:        
                    self.get_group_similarity(snap1, group1, snap2, group2)
        
        return self.similarities[key]
    
    def __calculate_snapshot_matching(self, snap1, snap2):
        """
        Inner function that calculates the matching between two snapshots.
                
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :return: Matching of two snapshots.
        :rtype: int

        """
        if (snap1, snap2) in self.matchings:
            return self.__sum_matching(snap1, snap2)
        
        similarities = self.__calculate_all_group_similarities(snap1, snap2)
        
        g = nx.Graph()
        for (group1, group2), sim in similarities.items():
            g.add_edge(f'{snap1}_{group1}', f'{snap2}_{group2}', weight=sim)
         
         
        edges = []
        for (group1, group2) in nx.max_weight_matching(g):
            if group1.startswith(snap1):
                edge = (group1.split("_")[1], group2.split("_")[1], g.edges[(group1, group2)]['weight'])
            else:
                edge = (group2.split("_")[1], group1.split("_")[1], g.edges[(group1, group2)]['weight'])
            edges += [edge]
        self.matchings[(snap1, snap2)] = edges
         
        return self.__sum_matching(snap1, snap2) 
    
    def get_snapshots(self):
        """
        Return the ids of snapshots.
        
        :return: List of ids of snapshots.
        :rtype: list

        """
        return self.groups.keys()
    
    def get_snapshot_similarity(self, snap1, snap2, groups=False):
        """
        Return the similarity between two snapshots.
        
        :param snap1: ID of 1st snapshot
        :type snap1: str
        :param snap2: ID of 2nd snapshot
        :type snap2: str
        :param groups: Whether return or not the groups of the matching, defaults to False
        :type groups: bool, optional
        :raises ValueError: If snapshot ID not snapshot IDs.
        :return: If groups=False, returns similarity of two snapshots. If groups=True, returns (sim, matching), i.e. detailed similarities of matching.
        :rtype: tuple

        """
        if snap1 not in self.groups:
            raise ValueError(f"Snapshot {snap1} not in snapshot ids")
        if snap2 not in self.groups:
            raise ValueError(f"Snapshot {snap2} not in snapshot ids")
        
        if (snap1, snap2) not in self.matchings:
            self.__calculate_snapshot_matching(snap1, snap2)
        
        if groups:
            return (self.__sum_matching(snap1, snap2), self.matchings[(snap1, snap2)])
        return self.__sum_matching(snap1, snap2) 
    
    
    def get_snapshot_evolution(self):
        """
        Return the vector representing the evolution of the entire snapshot sequence.
        
        :return: Evolution of snapshot sequence.
        :rtype: list

        """
        out = []
        keys = sorted(self.groups.keys())
        for i in range(0, len(keys)-1):
            out.append((keys[i], keys[i+1], self.get_snapshot_similarity(keys[i], keys[i+1], False)))
        return out
    
    
    def __calculate_group_similarity(self, group1, group2):
        """
        Inner function that calculates jaccard of two groups.
        
        :param group1: 1st group.
        :type group1: set
        :param group2: 2nd group.
        :type group2: set
        :return: Jaccard score of two groups.
        :rtype: int

        """
        return len(group1 & group2) / len(group1 | group2)
    
    def __get_group_concentration(self, groups, tau):
        """
        Inner function that calculates number of groups required to cover tau concentration.
        
        :param groups: Member distribution of current group.
        :type groups: list
        :param tau: Threshold for concentration.
        :type tau: double
        :return: Number of groups required to cover tau concentration.
        :rtype: int

        """
        s, no = 0, 0
        for g in groups:
            s += g[1]
            no += 1
            if s > tau:
                break
        return no
    
    def get_groups(self, snap):
        """
        Return the ids of groups inside a specific snapshot.
        
        :param snap: ID of snapshot.
        :type snap: str
        :raises ValueError: ID not in snapshot IDs.
        :return: List of ids of groups.
        :rtype: list

        """
        if snap not in self.groups:
            raise ValueError(f"Snapshot {snap} not in snapshot ids")
        return self.groups[snap].keys()    
    

    
    def get_group_similarity(self, snap1, group1, snap2, group2):
        """
        Return the jaccard similarity of two groups.
        
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param group1: ID of 1st group.
        :type group1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :param group2: ID of 2nd group.
        :type group2: str
        :raises ValueError: ID not in list of IDs.
        :return: Similarity of the two groups (jaccard).
        :rtype: int

        """
        if snap1 not in self.groups:
            raise ValueError(f"Snapshot {snap1} not in snapshot ids")
        if snap2 not in self.groups:
            raise ValueError(f"Snapshot {snap2} not in snapshot ids")
        if group1 not in self.groups[snap1]:
            raise ValueError(f"Group {group1} not in snapshot {snap1}")
        if group2 not in self.groups[snap2]:
            raise ValueError(f"Group {group2} not in snapshot {snap2}")
            
        if (snap1, snap2) not in self.similarities:
            self.similarities[(snap1, snap2)] = {}
        if (group1, group2) not in self.similarities[(snap1, snap2)]:
            g1 = self.groups[snap1][group1]
            
            g2 = self.groups[snap2][group2]
            
            self.similarities[(snap1, snap2)][(group1, group2)] = self.__calculate_group_similarity(g1, g2)
            
        return self.similarities[(snap1, snap2)][(group1, group2)]

 
    def get_group_evolution(self, snap1, group1, snap2, tau=0.8, alpha=1, beta=3):
        """
        Return the evolution of a group.
        
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param group1: ID of 1st group.
        :type group1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :param tau: Percentage collected, defaults to 0.8
        :type tau: float, optional
        :param alpha: Parameter for decision, defaults to 1
        :type alpha: int, optional
        :param beta: Parameter for decision, defaults to 3
        :type beta: int, optional
        :raises ValueError: ID not in IDs.
        :return: (status, related_groups): Status, i.e 0-> Similar, 1->Split, 2-> Diffused.  Groups, percentage in each group.
        :rtype: tuple

        """
        if snap1 not in self.groups:
            raise ValueError(f"Snapshot {snap1} not in snapshot ids")
        if snap2 not in self.groups:
            raise ValueError(f"Snapshot {snap2} not in snapshot ids")
        if group1 not in self.groups[snap1]:
            raise ValueError(f"Group {group1} not in snapshot {snap1}")
        
        c = Counter()
        for member in self.groups[snap1][group1]:
            if snap2 in self.inv_index[member]:
                c.update([self.inv_index[member][snap2]])
            else:
                c.update([-1])
                
        total = sum(cc for _,cc in c.items())
        related_groups = sorted(c.items(), key=lambda x: -x[1])
        related_groups = [(g, val/total) for g, val in related_groups]

        no_groups = self.__get_group_concentration(related_groups, tau)
        
        if no_groups < alpha:
            status = 0  # similar group
        elif no_groups < beta:
            status = 1  # split into 3 groups
        else:
            status = 2   # diffused
        return (status, related_groups)
    
        
    def get_group_provenance(self, snap1, group1, snap2, tau=0.8, alpha=1, beta=3):
        """
        Return the provenance of a group.
        
        :param snap1: ID of 1st snapshot.
        :type snap1: str
        :param group1: ID of 1st group.
        :type group1: str
        :param snap2: ID of 2nd snapshot.
        :type snap2: str
        :param tau: Percentage collected, defaults to 0.8
        :type tau: float, optional
        :param alpha: Parameter for decision, defaults to 1
        :type alpha: int, optional
        :param beta: Parameter for decision, defaults to 3
        :type beta: int, optional
        :raises ValueError: ID not in IDs.
        :return: (status, related_groups): Status, i.e 0-> Similar, 1->Split, 2-> Diffused.  Groups, percentage in each group.
        :rtype: tuple

        """        
        return self.get_group_evolution(snap1, group1, snap2, tau, alpha, beta)

            
    def get_members(self, snap, group):
        """
        Return the ids of members inside a specific group of a specific snapshot.
        
        :param snap: ID of snapshot.
        :type snap: str
        :param group: ID of group.
        :type group: str
        :raises ValueError: ID not in IDs.
        :return: List of ids of members.
        :rtype: list

        """
        if snap not in self.groups:
            raise ValueError(f"Snapshot {snap} not in snapshot ids")
        if group not in self.groups[snap]:
            raise ValueError(f"Group {group} not in snapshot {snap}")      
        return self.groups[snap][group]
    
    def get_groups_of_member(self, member):
        """
        Return the ids of (snapshot,group) that a member belongs to.
        
        :param member: ID of member.
        :type member: str
        :raises ValueError: Member ID not in IDs.
        :return: List of ids of (snapshot,group).
        :rtype: list

        """
        if member not in self.inv_index:
            raise ValueError(f"Member {member} not in member ids")
        return self.inv_index[member].items()
    

    def get_member_evolution(self, member):
        """
        Return the vector representing the evolution of the required member.
        
        :param member: ID of member.
        :type member: str
        :raises ValueError: Member ID not in member IDs.
        :return: Evolution of member.
        :rtype: list

        """
        if member not in self.inv_index:
            raise ValueError(f"Member {member} not in member ids")
            
        out = []
        keys = list(self.groups.keys())
        for i in range(0, len(keys)-1):
            if keys[i] in self.inv_index[member] and keys[i+1] in self.inv_index[member]:
                sim = self.get_group_similarity(keys[i], self.inv_index[member][keys[i]],
                                                keys[i+1], self.inv_index[member][keys[i+1]])
            else:
                sim = 0
            out.append((keys[i], keys[i+1], sim))
        return out
    
    
    def get_member_rules(self, members=None, min_support=0.07, min_threshold=1, metric='lift'):
        """
        Return rules from frequent subgroups mining.
        
        :param members: List of member ids to use for group filtering. If None is given, then all groups are used, defaults to None
        :type members: list, optional
        :param min_support: A float between 0 and 1 for minumum support of the itemsets returned, defaults to 0.07
        :type min_support: float, optional
        :param min_threshold: Minimal threshold for the evaluation metric, via the `metric` parameter, to decide whether a candidate rule is of interest, defaults to 1
        :type min_threshold: float, optional
        :param metric: Metric to use: 'lift', 'support', 'confidence', 'leverage', 'conviction', defaults to 'lift'
        :type metric: str, optional
        :return: pandas DataFrame with columns "antecedents" and "consequents"
              that store itemsets, plus the scoring metric columns:
              "antecedent support", "consequent support",
              "support", "confidence", "lift",
              "leverage", "conviction"
              of all rules for which
              metric(rule) >= min_threshold.
              Each entry in the "antecedents" and "consequents" columns are
              of type `frozenset`, which is a Python built-in type that
              behaves similarly to sets except that it is immutable
              (For more info, see
              https://docs.python.org/3.6/library/stdtypes.html#frozenset).
        :rtype: DataFrame

        """
        q_mem = members[0]
        no = len(self.inv_index[q_mem])
        
        d = {}
        for mem, gr in self.inv_index.items():
            if mem == q_mem:
                continue
            d[(mem,)] = len(gr)
            d[(q_mem, mem)] = len(gr)
        d[(q_mem,)] = no
            
        d2 = {}    
        for mem in d:
            d2[frozenset(mem)] = d[mem] / no
        
        df = pd.DataFrame({'support': d2.values(), 'itemsets': d2.keys()})
        df = df.loc[df.support > min_support]
        
        frequent_itemsets = df
        #create rules of A->B ONLY with confidence
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        
        #Filter and keep only those in members
        final_s = pd.Series([False]*rules.shape[0])
        for mem in members:
            s = rules.antecedents.apply(lambda x: mem in x)
            final_s = final_s | s
        rules = rules[s]
        
        return rules
        


class Graph():
    """Class for visualizing Change Detector methods."""
    
    def __init__(self):
        self.graph = ipycytoscape.CytoscapeWidget()
        self.graph.set_layout(name='preset')
        self.cached_nodes = set()
        self.step = 200
        
        self.colors = ['red', 'cyan', 'blue', 'darkblue', 'lightblue', 'orange', 'purple', 'brown',
                       'yellow', 'maroon', 'lime', 'green', 'magenta', 'olive', 'pink', 'aquamarine']
        
        style_colors =  [{'selector': f'.{c}', 'style': {'background-color': c, 'line-color': c}} for c in self.colors]
        
        style =[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                }
            },
            {
              'selector': "edge[label]",
              'css': {
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-x": "0px",
                "text-margin-y": "-20px"
              }
            },    
            {
                'selector': '.rectangle',
                'style': {
                    'shape': 'rectangle'
                }
            },      
            {
                'selector': '.rrectangle',
                'style': {
                    'shape': 'round rectangle'
                }
            },    
            {
                'selector': '.ellipse',
                'style': {
                    'shape': 'ellipse'
                }
            },            
            {
            'selector': '.member',
            'style': {
                'background-color': 'red',
                'line-color': 'red',
                'shape': 'ellipse',
                'height': '25px',
                'width': '25px',
                }
            },    
            {
            'selector': '.nomember',
            'style': {
                'shape': 'ellipse',
                'height': '25px',
                'width': '25px',
                }
            },
            {
            'selector': '.yomember',
            'style': {
                'shape': 'ellipse',
                'height': '50px',
                'width': '50px',
                'visible': 'false',
                }
            }          
        ]
        
        style += style_colors
        self.graph.set_style(style)
        
    def __random_position(self, coords, step=200):
        """
        Return coordinates of node in graph.
        
        :param coords: Dictionary of matrix coordinates with keys 'x' and 'y'
        :type coords: dict
        :param step: Step in increasing coords, defaults to 200
        :type step: int, optional
        :return: Dictionary of coordinates with keys 'x' and 'y'
        :rtype: dict

        """
        total_x, total_y = coords['x'], coords['y']
        return {'x': random.choice(range(total_x, total_x+step)),
                'y': random.choice(range(total_y, total_y+step))}         
        
    def init_graph(self, cd):
        """
        Create graph content.
        
        :param cd: Change_Detector object
        :type cd: class:`Change_Detector`
        :return: None
        :rtype: None

        """
        self.graph.graph.clear()
        
        snaps = cd.get_snapshots()
    
        nodes = []
        edges = []
        
        total_x = 0
        step = 200
        
        for i, snap in enumerate(snaps):
            nodes += [{"data": {"id": snap, "label":snap}, "grabbable": False, 'classes': 'rectangle'}]
            
            total_y = 0
            
            groups = cd.get_groups(snap)
            for j, group in enumerate(groups):
                
                members = cd.get_members(snap, group)
                
                nodes += [{"data": {"id": f'{snap}_{group}', "label": '{} ({})'.format(group, len(members)), 'parent': snap}, 
                           'position': {'x': total_x, 'y': total_y}, "grabbable": False, 
                           'classes': 'ellipse'}]
                
                total_y += step   
            total_x += step   
            
        j = {"nodes": nodes, 'edges': edges}
        
        self.initial_j = j
        self.graph.graph.add_graph_from_json(self.initial_j)
        

    def snapshot_evolution(self, cd):
        """
        Create graph content for Snapshot Evolution.
        
        :param cd: Change_Detector object
        :type cd: class:`Change_Detector`
        :return: None
        :rtype: None

        """
        self.__clean_nodes()
    
        evol = cd.get_snapshot_evolution()
    
        edges = [{'data': {'target': e[0], 'source': e[1], 
                           'weight': e[2], 'label': '{:.2f}'.format(e[2])}} for i, e in enumerate(evol)]
    
        j = {"nodes": [], 'edges': edges}
        self.graph.graph.add_graph_from_json(j)
        
        
    def snapshot_similarity(self, cd, snap1, snap2):
        """
        Create graph content for Snapshot Similarity between given snapshots.
        
        :param cd: Change_Detector object.
        :type cd: class:`Change_Detector`
        :param snap1: ID of Snapshot1
        :type snap1: str
        :param snap2: ID of Snapshot2
        :type snap2: str
        :return: None
        :rtype: None

        """
        self.__clean_nodes()
        
        matching = cd.get_snapshot_similarity(snap1, snap2, groups=True)
    
        edges = [{'data': {'target': f'{snap1}_{e[0]}', 'source': f'{snap2}_{e[1]}', 
                           'weight': e[2], 'label': '{:.2f}'.format(e[2])}} for e in matching[1]]
    
        j = {"nodes": [], 'edges': edges}
        self.graph.graph.add_graph_from_json(j)      
        
        
        
    def group_evolution(self, cd, snap1, group1, snap2, tau, alpha, beta): 
        """
        Create graph content for Group Evolution for given arguments.
        
        :param cd: Change_Detector object.
        :type cd: class:`Change_Detector`
        :param snap1: ID of Snapshot1.
        :type snap1: str
        :param group1: ID of Group1.
        :type group1: str
        :param snap2: ID of Snapshot2.
        :type snap2: str
        :param tau: Percentage collected.
        :type tau: float
        :param alpha: Parameter for decision.
        :type alpha: int
        :param beta: Parameter for decision.
        :type beta: int
        :return: None
        :rtype: None

        """
        self.__clean_nodes()
        
        evol = dict(cd.get_group_evolution(snap1, group1, snap2, tau, alpha, beta)[1])
        if -1 in evol:
            evol.pop(-1)
        edges = [{'data': {'target': f'{snap1}_{group1}', 'source': f'{snap2}_{e}', 
                           'weight': evol[e], 'label': '{:.2f}'.format(evol[e])}} for e in evol]
        
        
        lcolors = {e: self.colors[i] for i, e in enumerate(evol)}
        lcolors[None] = 'gray'
        
        parent_ids = set(['{}_{}'.format(snap2, e) for e in evol] + ['{}_{}'.format(snap1, group1)])
        parent_nodes = {}
        for i, node in enumerate(self.graph.graph.nodes):
            if node.data['id'] in parent_ids:
                parent_nodes[node.data['id']] = node
        
        
        nodes = []
        for m in cd.groups[snap1][group1]:
            group2 = cd.inv_index[m][snap2] if snap2 in cd.inv_index[m] else None
            
            par1 = '{}_{}'.format(snap1, group1)
            nodes += [{"data": {"id": '{}_{}'.format(par1, m),"label": m, 'parent': par1}, 
                       "grabbable": True, 'classes': lcolors[group2],
                       'position': self.__random_position(parent_nodes[par1].position, self.step//2)
                       }]

            if group2 is not None:
                par2 = '{}_{}'.format(snap2, group2)
                nodes += [{"data": {"id": '{}_{}'.format(par2, m),"label": m, 'parent': par2 }, 
                           "grabbable": True, 'classes': lcolors[group2],
                           'position': self.__random_position(parent_nodes[par2].position, self.step//2)
                           }]
            
        j = {"nodes": nodes, 'edges': edges}
        self.graph.graph.add_graph_from_json(j)      


    def group_provenance(self, cd, snap1, group1, snap2, tau, alpha, beta): 
        """
        Create graph content for Group Provenance for given arguments.
        
        :param cd: Change_Detector object.
        :type cd: class:`Change_Detector`
        :param snap1: ID of Snapshot1.
        :type snap1: str
        :param group1: ID of Group1.
        :type group1: str
        :param snap2: ID of Snapshot2.
        :type snap2: str
        :param tau: Percentage collected.
        :type tau: float
        :param alpha: Parameter for decision.
        :type alpha: int
        :param beta: Parameter for decision.
        :type beta: int
        :return: None
        :rtype: None

        """        
        self.__clean_nodes()
        
        evol = dict(cd.get_group_provenance(snap1, group1, snap2, tau, alpha, beta)[1])
        if -1 in evol:
            evol.pop(-1)        
        edges = [{'data': {'target': f'{snap1}_{group1}', 'source': f'{snap2}_{e}', 
                           'weight': evol[e], 'label': '{:.2f}'.format(evol[e])}} for e in evol]
    
        lcolors = {e: self.colors[i] for i, e in enumerate(evol)}
        lcolors[None] = 'gray'
        
        parent_ids = set(['{}_{}'.format(snap2, e) for e in evol] + ['{}_{}'.format(snap1, group1)])
        parent_nodes = {}
        for i, node in enumerate(self.graph.graph.nodes):
            if node.data['id'] in parent_ids:
                parent_nodes[node.data['id']] = node
        
        
        nodes = []
        for m in cd.groups[snap1][group1]:
            group2 = cd.inv_index[m][snap2] if snap2 in cd.inv_index[m] else None
            par1 = '{}_{}'.format(snap1, group1)
            nodes += [{"data": {"id": '{}_{}'.format(par1, m),"label": m, 'parent': par1}, 
                       "grabbable": True, 'classes': lcolors[group2],
                       'position': self.__random_position(parent_nodes[par1].position, self.step//2)
                       }]

            if group2 is not None:
                par2 = '{}_{}'.format(snap2, group2)
                nodes += [{"data": {"id": '{}_{}'.format(par2, m),"label": m, 'parent': par2 }, 
                           "grabbable": True, 'classes': lcolors[group2],
                           'position': self.__random_position(parent_nodes[par2].position, self.step//2)
                           }]
            
        j = {"nodes": nodes, 'edges': edges}    
    
        self.graph.graph.add_graph_from_json(j)           
        
        
    def group_similarity(self, cd, snap1, group1, snap2, group2): 
        """
        Create graph content for Group Similarity for given arguments.
        
        :param cd: Change_Detector object.
        :type cd: class:`Change_Detector`
        :param snap1: ID of Snapshot1.
        :type snap1: str
        :param group1: ID of Group1.
        :type group1: str
        :param snap2: ID of Snapshot2.
        :type snap2: str
        :param group2: ID of Group2.
        :type group2: str
        :return: None
        :rtype: None

        """
        self.__clean_nodes()
        
        sim = cd.get_group_similarity(snap1, group1, snap2, group2)
        
        g1 = set(cd.get_members(snap1, group1))
        g2 = set(cd.get_members(snap2, group2))
        common = g1 & g2
        
        par1 = '{}_{}'.format(snap1, group1)
        par2 = '{}_{}'.format(snap2, group2)
        
        parent_nodes = {}
        for i, node in enumerate(self.graph.graph.nodes):
            if node.data['id'] in [par1, par2]:
                parent_nodes[node.data['id']] = node
        
        
        nodes = []
        for m in g1:
            classes = 'member' if m in common else 'nomember'
            nodes += [{"data": {"id": '{}_{}'.format(par1, m),"label": m, 'parent': par1}, 
                       "grabbable": True, 'position': self.__random_position(parent_nodes[par1].position, self.step//2),
                       'classes': classes}]
        for m in g2:
            classes = 'member' if m in common else 'nomember'
            nodes += [{"data": {"id": '{}_{}'.format(par2, m),"label": m, 'parent': par2}, 
                       "grabbable": True, 'position': self.__random_position(parent_nodes[par2].position, self.step//2),
                       'classes': classes}]            
        
        
        edges = [{'data': {'target': par1, 'source': par2, 
                           'weight': sim, 'label': '{:.2f}'.format(sim)}}]
    
        j = {"nodes": nodes, 'edges': edges}
        self.graph.graph.add_graph_from_json(j)         
        
        
    def member_evolution(self, cd, member):
        """
        Create graph content for Member Evolution for given member.
        
        :param cd: Change_Detector object.
        :type cd: class:`Change_Detector`
        :param member: ID of Member.
        :type member: str
        :return: None
        :rtype: None

        """
        self.__clean_nodes()        
        
        evol = cd.get_member_evolution(member)
        groups = dict(cd.get_groups_of_member(member))

        parent_ids = set([f'{s}_{g}' for (s, g) in groups.items()])

        parent_nodes = {}
        for i, node in enumerate(self.graph.graph.nodes):
            if node.data['id'] in parent_ids:
                parent_nodes[node.data['id']] = node
             
        nodes, edges = {}, []
        for e in evol:
            if e[0] not in groups or e[1] not in groups:
                continue
            
            edges += [{'data': {'target': '{}_{}_{}'.format(e[0], groups[e[0]], member),
                                'source': '{}_{}_{}'.format(e[1], groups[e[1]], member),
                                'weight': e[2], 'label': '{:.2f}'.format(e[2])}}]

            parent_id = '{}_{}'.format(e[0], groups[e[0]])
            nid = f'{parent_id}_{member}'
            nodes[nid] = {"data": {"id": nid,
                            "label": member, 'parent': parent_id}, 
                          "grabbable": False, 'position': parent_nodes[parent_id].position,
                          'classes': 'member'}
            self.cached_nodes.add(nid)

            parent_id = '{}_{}'.format(e[1], groups[e[1]])
            nid = f'{parent_id}_{member}'
            nodes[nid] = {"data": {"id": nid,
                            "label": member, 'parent': parent_id}, 
                          "grabbable": False, 'position': parent_nodes[parent_id].position,
                          'classes': 'member'}
            self.cached_nodes.add(nid)
        nodes = nodes.values()

        j = {"nodes": nodes, 'edges': edges}
        self.graph.graph.add_graph_from_json(j)
        
        
    def __clean_nodes(self):
        self.graph.graph.clear()
        self.graph.graph.add_graph_from_json(self.initial_j)
        