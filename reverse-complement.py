seq = "ATCGATCG"

complements = {
    'A' : 'T',
    'T' : 'A',
    'G' : 'C',
    'C' : 'G'
}

dna = ""

for letter in seq:
    dna += complements[letter]

reverse_complement = dna[::-1]
rna = reverse_complement.replace('T', 'U')

print(f"DNA: {seq}, RNA: {rna}")
print(f"reverse DNA: {reverse_complement}")