"""SIGEF - Fase 2 | Valentina Gomez
BFS, DFS y A* para priorizar tareas de mantenimiento.
Costos y riesgos simulados con fines academicos.
Python 3.9+; sin dependencias externas.
"""
from collections import deque
from heapq import heappush, heappop
from time import perf_counter
import statistics
import tracemalloc

# (codigo, vehiculo, actividad, costo_base, riesgo)
TASKS = [
    ("T1","TLW 411","Aceite",120,8), ("T2","TLW 411","Direccion",180,10),
    ("T3","TLW 458","Llantas",350,9), ("T4","TLW 458","Frenos",220,12),
    ("T5","TLW 271","Aceite",120,7), ("T6","TLW 271","Suspension",200,8),
    ("T7","TLV 715","Motor",400,15), ("T8","TLV 715","Caja",300,11),
    ("T9","TLW 649","Aire acondicionado",160,5),
    ("T10","TLW 649","Sistema electrico",190,6),
]
N=len(TASKS); START=0; GOAL=(1<<N)-1; LAMBDA=1

def pending(state):
    return [i for i in range(N) if not state & (1<<i)]

def successors(state):
    for i in pending(state):
        yield i, state | (1<<i)

def step_cost(next_state, task_index):
    base=TASKS[task_index][3]
    risk=sum(TASKS[j][4] for j in pending(next_state))
    return base + LAMBDA*risk

def heuristic(state):
    return sum(TASKS[i][3] for i in pending(state))

def reconstruct(parent,state):
    path=[]
    while state!=START:
        previous,action=parent[state]
        path.append(action); state=previous
    return list(reversed(path))

def route_cost(path):
    state=START; total=0
    for i in path:
        nxt=state|(1<<i); total+=step_cost(nxt,i); state=nxt
    return total

def bfs():
    queue=deque([START]); parent={}; visited={START}; expanded=0
    while queue:
        state=queue.popleft()
        if state==GOAL:
            path=reconstruct(parent,state)
            return path,route_cost(path),expanded
        expanded+=1
        for i,nxt in successors(state):
            if nxt not in visited:
                visited.add(nxt); parent[nxt]=(state,i); queue.append(nxt)
    raise RuntimeError("BFS sin solucion")

def dfs():
    stack=[START]; parent={}; visited={START}; expanded=0
    while stack:
        state=stack.pop()
        if state==GOAL:
            path=reconstruct(parent,state)
            return path,route_cost(path),expanded
        expanded+=1
        for i,nxt in reversed(list(successors(state))):
            if nxt not in visited:
                visited.add(nxt); parent[nxt]=(state,i); stack.append(nxt)
    raise RuntimeError("DFS sin solucion")

def astar():
    counter=0; frontier=[(heuristic(START),0,counter,START)]
    best_g={START:0}; parent={}; expanded=0
    while frontier:
        f,g,_,state=heappop(frontier)
        if g!=best_g.get(state): continue
        if state==GOAL:
            path=reconstruct(parent,state)
            return path,g,expanded
        expanded+=1
        for i,nxt in successors(state):
            new_g=g+step_cost(nxt,i)
            if new_g<best_g.get(nxt,float("inf")):
                best_g[nxt]=new_g; parent[nxt]=(state,i); counter+=1
                heappush(frontier,(new_g+heuristic(nxt),new_g,counter,nxt))
    raise RuntimeError("A* sin solucion")

def benchmark(fn,repetitions=30):
    times=[]; peaks=[]; answer=None
    for _ in range(repetitions):
        tracemalloc.start(); t0=perf_counter()
        answer=fn(); elapsed=perf_counter()-t0
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        times.append(elapsed); peaks.append(peak)
    return answer,statistics.median(times),statistics.median(peaks)

def main():
    print("SIGEF | escenario academico simulado | 30 ejecuciones")
    for name,fn in (("BFS",bfs),("DFS",dfs),("A*",astar)):
        (path,cost,nodes),tm,mem=benchmark(fn)
        print(f"{name}: nodos={nodes}, mediana_s={tm:.8f}, costo={cost}, memoria_B={int(mem)}")
        print("Ruta: "+" -> ".join(TASKS[i][0] for i in path))
if __name__=="__main__": main()
