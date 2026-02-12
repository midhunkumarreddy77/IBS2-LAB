from Bio.Seq import Seq

def transcription1(dna):
    transcription = ""

    for base in dna:
        match base:
            case "T":
                transcription+="U"
            case "A":
                transcription+=base
            case "G":
                transcription += base
            case "C":
                transcription += base

    return transcription


def translation1(rna):
    codon_table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',  # STOP
        'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',  # STOP/Trp
    }

    protein = ""
    for i in range(0, len(rna), 3):
        codon = rna[i:i+3]
        amino_acid = codon_table.get(codon, 'X')
        if amino_acid == '*':
            break
        protein += amino_acid

    return protein


print(f"DNA: ATGACAGTAAT")
print(f"Transcription using user defined function: {transcription1('ATGACAGTAAT')}")
print(f"Transcription using Biopython: {Seq('ATGACAGTAAT').transcribe()}")
print(f"Translation using Biopython: {Seq('ATGACAGTATAAATT').translate(stop_symbol='*', to_stop=True)}")
print(f"Translation using user defined function: Protein: {translation1('ATGACAGTATAAATT')}")