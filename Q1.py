from collections import defaultdict
from nltk import Tree
grammar_rules={
    'S': [('NP', 'VP')],
    'VP': [('V', 'NP'), ('VP', 'PP')],
    'NP': [('NP', 'PP'), ('John',), ('Mary',), ('Denver',)],
    'PP': [('P', 'NP')],
    'V': [('called',)],
    'P': [('from',)],
}

tree1 = (
    'S',
    ('NP', ('John',)),
    ('VP',
        ('V', ('called',)),
        ('NP',
            ('NP', ('Mary',)),
            ('PP',
                ('P', ('from',)),
                ('NP', ('Denver',))
            )
        )
    )
)

tree2 = (
    'S',
    ('NP', ('John',)),
    ('VP',
        ('VP',
            ('V', ('called',)),
            ('NP', ('Mary',))
        ),
        ('PP',
            ('P', ('from',)),
            ('NP', ('Denver',))
        )
    )
)

grammar_rules=defaultdict(lambda: defaultdict(float))
grammar_rules_reversed=defaultdict(tuple)

def compute_probabilities(tree):
  if (len(tree)==2):
    grammar_rules[tree[0]][tree[1]]+=1
  else:
    n_t=tree[0]
    left_nt=tree[1][0]
    right_nt=tree[2][0]
    grammar_rules[n_t][(left_nt,right_nt)]+=1
    compute_probabilities(tree[1])
    compute_probabilities(tree[2])


compute_probabilities(tree1)
compute_probabilities(tree2)
for n_t,rules in grammar_rules.items():
  total_count=sum(rules.values())
  for rule in rules:
    grammar_rules[n_t][rule]=grammar_rules[n_t][rule]/total_count
    grammar_rules_reversed[rule]=(n_t,grammar_rules[n_t][rule])

sentence = ["John", "called", "Mary", "from", "Denver"]
length=len(sentence)

table=[[set() for _ in range (length) ]for _ in range(length)]
for i in range(length):
  n_t=grammar_rules_reversed[(sentence[i],)]
  table[i][i].add((n_t[0],sentence[i],n_t[1]))

for span_length in range(2,length+1):
  for i in range(length-span_length+1):
    j=i+span_length-1
    for k in range(i,j):
      for left in table[i][k]:
        for right in table[k+1][j]:
          left_n_t,prob_left=left[0],left[-1]
          right_n_t,prob_right=right[0],right[-1]
          if ((left_n_t,right_n_t) in grammar_rules_reversed):
            n_t=grammar_rules_reversed[(left_n_t,right_n_t)]
            prob=prob_left*prob_right*n_t[1]
            table[i][j].add((n_t[0],left,right,prob))

top_cell=table[0][length-1]
def buildtree(tree):
  if (len(tree)==3):
    return Tree(tree[0],[tree[1]])
  else:
    return Tree(tree[0],[buildtree(tree[1]),buildtree(tree[2])])

for tree in top_cell:
  tree=buildtree(tree)
  tree.pretty_print()

best_tree=max(top_cell,key=lambda k:k[-1])
tree=buildtree(best_tree)
tree.pretty_print()
print(best_tree[-1])