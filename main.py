from exh import *
from itertools import chain, combinations, permutations
import string
import numpy as np
import functools
import sys
import re

### Parameters

conj_num = 2 # Maximal number of arguments for a connective in a sentence
prop_num = 2 # Number of atomic propositions generated

num_iter = 1000 # Number of iterations (of prior assignment)

# prior_type = "flat"
prior_type = "random"


# epistemic = True # An epistemic game when true
epistemic = False

deletion_alts = True # When true, atomic propositions are generated as alternatives (regardless of connective identity)
# deletion_alts = False

# subdomains = True # When true, subdomain alts where a connective takes fewer arguments than conj_num are generated
subdomains = False

smallest_subdomains = 2 # minimal cardinality of set of arguments for subdomain alts

arguments_as_set = True # When True, a connective takes a set as an argument
# arguments_as_set = False # When False, connectives are binary and can take more than two arguments through recursion

alts_to_ignore = "props" # only atomic propisitions are ignored when stability is checked
# alts_to_ignore = "props+subdomains" # atomic propisitions and subdomain alternatives are ignored for stability
# alts_to_ignore = "none" # no propisitions are ignored when stability is checked

# non_commutative_cons = True # When True, Non-commutative connectives are generated
non_commutative_cons = False

# trivial_cons = True # When True, trivial connectives are generated
trivial_cons = False

###

def ps(iterable, min_len=1, subtract_from_max_len=0, large_first=False, all_orderings=False):
    s = list(iterable)
    set_sizes = range(min_len, len(s) + 1 - subtract_from_max_len)
    if large_first:
        set_sizes = reversed(set_sizes)
    if all_orderings:
        ch = chain.from_iterable(permutations(s, r) for r in set_sizes)
    else:
        ch = chain.from_iterable(combinations(s, r) for r in set_sizes)
    return (list(i) for i in ch)

def gen_alts(con_dict, lang, props, conjs):
    if subdomains:
        argslist = ps(props, smallest_subdomains, len(props) - len(conjs), large_first=True, all_orderings=non_commutative_cons)
    else:
        argslist = ps(props, len(conjs), len(props) - len(conjs), large_first=True, all_orderings=non_commutative_cons)
    alts = {}
    for args in argslist:
        args_names = list(map(lambda x: re.sub(r"[$]", "", str(x)), args))
        for con in lang:
            if arguments_as_set:
                set_name = ",".join(args_names)
                alts[f"{con}({set_name})"] = con_dict[con](args)
            else:
                alt_name = functools.reduce(lambda x, y: f"{x} {con} {y}", args_names)
                alts[alt_name] = functools.reduce(lambda x, y: con_dict[con]([x, y]), args)
    if deletion_alts:
        for i in range(len(props)):
            alts[re.sub(r"[$]", "", str(props[i]))] = props[i]
    return alts

def states(lang):
    st_num = np.shape(lang)[1]
    if epistemic:
        st = list(ps(range(st_num)))
    else:
        st = [[i] for i in range(st_num)]
    stbool = [[j in i for j in range(st_num)] for i in st]
    return stbool

def priors(lang, prior_type):
    rand_prior_dist = np.random.dirichlet(1 * np.ones(np.shape(states(lang))[0]))
    if prior_type == "flat":
        state_priors = [1 / np.shape(states(lang))[0]] * np.shape(states(lang))[0]  # flat priors
    elif prior_type == "random":
        state_priors = rand_prior_dist  # random priors
    else:
        raise Exception("unknown prior type")
    return state_priors

def amax_unless_zero(a):
    if np.amax(a) == 0:
        return np.array([False] * len(a))
    else:
        return a == np.amax(a)

def naive_hearer(lang, state_priors):
    lang_prob_dist = []
    stbool = states(lang)
    for message in lang:
        comp_states = [all(np.logical_or(np.invert(st), message)) for st in stbool]
        masked_state_priors = np.where(comp_states, state_priors, 0)
        # print(masked_state_priors)
        prob_given_message = masked_state_priors / sum(masked_state_priors)
        lang_prob_dist.append(prob_given_message)
    return np.nan_to_num(lang_prob_dist)

def rational_speaker(lang, level, state_priors):
    if level == 1:
        hearer_prob_dist = naive_hearer(lang, state_priors)
    elif level > 1:
        hearer_prob_dist = rational_hearer(lang, level - 1, state_priors)
    speaker_choices = list(map(amax_unless_zero, np.transpose(hearer_prob_dist)))
    speaker_choices = [list(map(int, i)) for i in speaker_choices]
    speaker_choices = np.transpose(np.nan_to_num(np.transpose(speaker_choices) / np.sum(np.array(speaker_choices), 1)))
    return np.array(speaker_choices)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f"results.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    old_stdout = sys.stdout
    sys.stdout = Logger()
    print(f'Parameter setting:')
    print(f'conj_num = {conj_num}')
    print(f'prop_num = {prop_num}')
    print(f'num_iter = {num_iter}')
    print(f'prior_type = {prior_type}')
    print(f'epistemic = {epistemic}')
    print(f'subdomains = {subdomains}')
    print(f'smallest_subdomains = {smallest_subdomains}')
    print(f'alts_to_ignore = {alts_to_ignore}')
    print(f'arguments_as_set = {arguments_as_set}')
    print(f'non_commutative_cons = {non_commutative_cons}')
    print(f'trivial_cons = {trivial_cons}')
    print("\n----------------------------\n")

    props = [Pred(index=key + 1, name=f"{val}")
             for key, val in enumerate(string.ascii_lowercase[:prop_num])]

    conjs = props[:conj_num]

    # print(props)
    # print(props[:conj_num])

    con_dict = {}
    con_dict["AND"] = lambda lst: functools.reduce(lambda x, y: x & y, lst)
    con_dict["OR"] = lambda lst: functools.reduce(lambda x, y: x | y, lst)
    con_dict["NAND"] = lambda lst: ~con_dict["AND"](lst)
    con_dict["NOR"] = lambda lst: ~con_dict["OR"](lst)
    con_dict["XOR"] = lambda lst: con_dict["OR"](lst) & ~con_dict["AND"](lst)
    con_dict["IFF"] = lambda lst: con_dict["AND"](lst) | ~con_dict["OR"](lst)

    if non_commutative_cons:

        if conj_num > 2 and arguments_as_set:
            raise Exception("No implementation for non-commutative connectives with more than 2 conjuncts when arguments are taken as a set")

        con_dict["ONLYL"] = lambda lst: lst[0] & ~lst[-1]
        con_dict["ONLYR"] = lambda lst: lst[-1] & ~lst[0]
        con_dict["->"] = lambda lst: ~lst[0] | lst[-1]
        con_dict["<-"] = lambda lst: ~lst[-1] | lst[0]
        # con_dict["L"] = lambda lst: lst[0]
        # con_dict["R"] = lambda lst: lst[-1]
        con_dict["NOTL"] = lambda lst: ~lst[0]
        con_dict["NOTR"] = lambda lst: ~lst[-1]

    if trivial_cons:
        con_dict["TAU"] = lambda lst: lst[0] | ~lst[0]
        con_dict["CONT"] = lambda lst: lst[0] & ~lst[0]

    prop_universe = Universe(fs=props)
    tv_table = np.flip(np.transpose(prop_universe.evaluate(*props)), 1)
    state_names = []
    for state in np.transpose(tv_table):
        state_name = []
        for key_prop, value_prop in enumerate(props):
            if state[key_prop]:
                state_name.append(re.sub(r"[$]" ,"" , str(value_prop)))
            else:
                state_name.append("~"+re.sub(r"[$]" ,"" , str(value_prop)))
        state_name = "&".join(state_name)
        state_names.append(state_name)
    if epistemic:
        state_names = [" U ".join(state_name) for state_name in ps(state_names)]
    print("Order of states in all tables below is as follows:")
    for key,val in enumerate(state_names):
        print(f"State {key} : {val}")
    print("\n----------------------------\n")

    speaker_choice_nums_table = []

    for lang in ps(con_dict):

        # print(lang)

        alts = gen_alts(con_dict, lang, props, conjs)
        pos_alts = list(alts.values())
        pos_alts_names = list(alts.keys())
        neg_alts = [~i for i in pos_alts]
        neg_alts_names = [f"~({i})" for i in pos_alts_names]

        tv_table_pos = np.flip(np.transpose(prop_universe.evaluate(*pos_alts)), 1)
        tv_table_neg = np.flip(np.transpose(prop_universe.evaluate(*neg_alts)), 1)
        # print(list(zip(pos_alts_names,tv_table_pos)))
        # print(list(zip(neg_alts_names,tv_table_neg)))

        if alts_to_ignore == "props":
            num_del = len(props)
        elif alts_to_ignore == "props+subdomains":
            num_del = len(alts) - len(lang)
        elif alts_to_ignore == "none":
            num_del = 0

        # print(lang)

        print("Language: ", lang)
        print("Alternatives (positive case): ", pos_alts_names)
        print("Alternatives (negative case): ", neg_alts_names)
        # print("Alternative meanings (positive case):", pos_alts)
        # print("Alternative meanings (negative case):", neg_alts)
        print("Alternatives checked for stability (positive case): ", pos_alts_names[:len(alts) - num_del])
        print("Alternatives checked for stability (negative case): ", neg_alts_names[:len(alts) - num_del])
        print("")

        speaker_choices = []

        for i in range(num_iter):
            speaker_state_priors = priors(tv_table_pos, prior_type)

            pos_speaker = rational_speaker(tv_table_pos, 1, speaker_state_priors)
            neg_speaker = rational_speaker(tv_table_neg, 1, speaker_state_priors)

            speaker_result = (
                np.transpose(pos_speaker)[:len(alts) - num_del],
                np.transpose(neg_speaker)[:len(alts) - num_del]
            )

            if not any(
                    all(
                        np.all(speaker_result[j] == old_result[j])
                        for j in range(len(old_result)))
                    for old_result in speaker_choices):
                speaker_choices.append(speaker_result)

        speaker_choice_num = len(speaker_choices)

        print("Number of different speaker tables:", speaker_choice_num)
        speaker_choice_nums_table.append([speaker_choice_num, lang])
        if speaker_choice_num == 1:
            print("The language is speaker STABLE in the simulation")
            print("")
            print("Optimal choices in positive sentences:\n", list(zip(pos_alts_names[:num_del],speaker_choices[0][0])))
            print("Optimal choices in negative sentences:\n", list(zip(pos_alts_names[:num_del],speaker_choices[0][1])))
        else:
            print("The language is speaker UNSTABLE in the simulation")
            print("")
            th = 4 # maximum number of tables to be shown for an unstable language
            if speaker_choice_num > th:
                print(f"Only {th} tables among the {speaker_choice_num} found are shown below\n")
            for i in range(min(speaker_choice_num, th)):
                print(f"Optimal choices in positive sentences in table {i+1}:\n",
                      list(zip(pos_alts_names[:num_del],speaker_choices[i][0])))
                print(f"Optimal choices in negative sentences in table {i+1}:\n",
                      list(zip(pos_alts_names[:num_del],speaker_choices[i][1])))
                print("")

        print("\n----------------------------\n")

    print("Sorting languages by number of possible speaker choices:")
    for i in sorted(speaker_choice_nums_table):
        print(i)
    print("")



    sys.stdout = old_stdout


if __name__ == '__main__':
    main()