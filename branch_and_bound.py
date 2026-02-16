import sys
MASS = {'G':57,'A':71,'S':87,'P':97,'V':99,'T':101,'C':103,
        'I':113,'L':113,'N':114,'D':115,'K':128,'Q':128,
        'E':129,'M':131,'H':137,'F':147,'R':156,'Y':163,'W':186}
def mass(peptide):
    return sum(MASS[aa] for aa in peptide)
def spectrum(peptide, cyclic=True):
    prefix = [0]
    for aa in peptide:
        prefix.append(prefix[-1] + MASS[aa])
    spec = [0]
    n = len(peptide)
    for i in range(n):
        for j in range(i+1, n+1):
            spec.append(prefix[j] - prefix[i])
            if cyclic and i>0 and j<n:
                spec.append(prefix[-1] - (prefix[j] - prefix[i]))
    return sorted(spec)
def consistent(peptide, spec):
    temp = spec.copy()
    for m in spectrum(peptide, cyclic=False):
        if m in temp:
            temp.remove(m)
        else:
            return False
    return True
def branch_and_bound(spec):
    target = max(spec)
    peptides = ['']
    results = []
    while peptides:
        new = []
        for p in peptides:
            for aa in MASS:
                x = p + aa
                m = mass(x)
                if m == target:
                    if spectrum(x) == sorted(spec):
                        results.append(x)
                elif m < target and consistent(x, spec):
                    new.append(x)
        peptides = new
    return results
def parse_spectrum(text):
    parts = text.replace(',', ' ').split()
    return sorted(int(x) for x in parts)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            spec = parse_spectrum(f.read())
    else:
        s = input("Paste spectrum (space/comma separated): ").strip()
        spec = parse_spectrum(s)
    results = branch_and_bound(spec)
    print(f"Found {len(results)} peptides")
    for p in results[:10]:
        print(" ", p)
    if len(results) > 10:
        print("  ...", len(results)-10, "more")