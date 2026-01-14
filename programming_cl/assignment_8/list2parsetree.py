import constituents 
'''markdown
This function accepts a list in this format:

[ ' S ' ,
[ ' NP ' ,
[ ' DT ' , ' The ' ],
[ ' NN ' , ' cat ' ]
],
[ ' VP ' ,
[ ' VB ' , ' chases ' ],
[ ' NP ' ,
[ ' DT ' , ' the ' ],
[ ' NN ' , ' mouse ' ]
]
],
[ ' PUNCT ' , ' . ' ]
]

and returns the root node of a constituency tree made of
Phrase and Token objects.
'''

def list2parsetree(input_list):
    intermediate_list = []
    final_phrase: constituents.Phrase

    if isinstance(input_list[1], str):
        return constituents.Token(input_list[0], input_list[1])
    else:
        for child in input_list[1:]:
            intermediate_list.append(list2parsetree(child))
        final_phrase = constituents.Phrase(input_list[0], intermediate_list) 
    
    return final_phrase