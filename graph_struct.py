# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:12:17 2019

@author: c53130a
"""

from collections import defaultdict 
  
class Graph: 
  
    def __init__(self,vertices): 
        self.V= vertices 
        self.graph= defaultdict(list)
        for i in range(self.V):
            self.graph[i+1] = []
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def getsub(self, n, d, visited):
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
            sub.extend(self.getsub(stk, d-1, visited))
        
        return sub