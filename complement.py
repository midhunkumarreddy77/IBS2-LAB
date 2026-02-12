from Bio.Seq import Seq

def complement1(seq):
    complement = ""

    for base in seq:
        match base:
            case "A":
                complement+="T"
            case "T":
                complement+="A"
            case "G":
                complement+="C"
            case "C":
                complement+="G"
            case _:
                print("Errror!")

    return complement

print("From user defined function:",complement1("ATGCGATCAT"))
print("From Biopython package:", Seq("ATGCGATCAT").complement())