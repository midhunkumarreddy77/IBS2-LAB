from collections import Counter
MASS={'G':57,'A':71,'S':87,'P':97,'V':99,'T':101,'C':103,
      'I':113,'L':113,'N':114,'D':115,'K':128,'Q':128,
      'E':129,'M':131,'H':137,'F':147,'R':156,'Y':163,'W':186}
def mass(p): return sum(MASS[a] for a in p)
def spectrum(p,cyclic=True):
    pre=[0]
    for a in p: pre.append(pre[-1]+MASS[a])
    s=[0]; n=len(p)
    for i in range(n):
        for j in range(i+1,n+1):
            s.append(pre[j]-pre[i])
            if cyclic and i>0 and j<n:
                s.append(pre[-1]-(pre[j]-pre[i]))
    return sorted(s)
def score(p,spec):
    t=spectrum(p)
    c=Counter(spec)
    sc=0
    for x in t:
        if c[x]>0:
            sc+=1
            c[x]-=1
    return sc
def trim(board,spec,N):
    board.sort(key=lambda x:score(x,spec),reverse=True)
    if len(board)<=N: return board
    cut=score(board[N-1],spec)
    return [p for p in board if score(p,spec)>=cut]
def leaderboard(spec,N):
    target=max(spec)
    board=['']
    best=''
    while board:
        new=[]
        for p in board:
            for a in MASS:
                x=p+a
                if mass(x)<=target:
                    new.append(x)
        board=[]
        for p in new:
            if mass(p)==target:
                if score(p,spec)>score(best,spec):
                    best=p
            else:
                board.append(p)
        board=trim(board,spec,N)
    return best
if __name__=="__main__":
    spec=list(map(int,input("Spectrum: ").split()))
    N=int(input("Leaderboard size: "))
    print("Best peptide:",leaderboard(spec,N))