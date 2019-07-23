from collections import defaultdict
  
class Graph: 
  
    def __init__(self,vertices): 
        self.V= vertices 
        self.graph= defaultdict(list)
        for i in range(self.V):
            self.graph[i+1] = []
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def bfs(self, n, d, visited):
        for i in n:
            visited[i-1] = True
        
        sub = []
        
        if d == 0:
            return n
        else:
            stk = []
            for i in n:
                stk.extend(self.graph[i])
            stk = list(set([x for x in stk if not visited[x-1]]))
            sub.extend(self.bfs(stk, d-1, visited))
        
        return sub

    def getsub(self, n, d):
        nodes = []
        for i in range(d):
            visited = [False]*len(self.graph)
            nodes.extend(self.bfs([n], i, visited))

        visited = [False] * len(self.graph)
        leaf = self.bfs([n], d, visited)

        sub = {x:self.graph[x] for x in nodes}
        for l in leaf:
            sub[l] = []

        return sub