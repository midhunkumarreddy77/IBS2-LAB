MASS = {'G':57,'A':71,'S':87,'P':97,'V':99,'T':101,'C':103,
        'I':113,'L':113,'N':114,'D':115,'K':128,'Q':128,
        'E':129,'M':131,'H':137,'F':147,'R':156,'Y':163,'W':186}
def mass(p):
    return sum(MASS[a] for a in p)
def spectrum(p):
    pre=[0]
    for a in p: pre.append(pre[-1]+MASS[a])
    s=[0]
    n=len(p)
    for i in range(n):
        for j in range(i+1,n+1):
            s.append(pre[j]-pre[i])
            if i>0 and j<n:
                s.append(pre[-1]-(pre[j]-pre[i]))
    return sorted(s)
def brute(spec):
    target=max(spec)
    peptides=['']
    ans=[]
    while peptides:
        new=[]
        for p in peptides:
            for a in MASS:
                x=p+a
                m=mass(x)
                if m==target:
                    if spectrum(x)==sorted(spec):
                        ans.append(x)
                elif m<target:
                    new.append(x)
        peptides=new
    return ans
if __name__=="__main__":
    spec=list(map(int,input("Spectrum: ").split()))
    r=brute(spec)
    print("Found",len(r),"peptides")
    for p in r[:10]: print(p)