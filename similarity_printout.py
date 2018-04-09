from util_kw import *

if __name__=='__main__':
    print ("\n\n **** Upper caste keywords and negative aspect ****")
    print_similarities_between_k_lists(negative_aspect, upper_caste_keywords)
    
    print ("\n\n **** Lower caste keywords and negative aspect ****")
    print_similarities_between_k_lists(negative_aspect, lower_caste_keywords)

    print ("\n\n **** Lower caste keywords and priviledge keywords ****")
    print_similarities_between_k_lists(lower_caste_keywords, priviledge_keywords)

    print ("\n\n **** Neutral keywords and priviledge keywords ****")
    print_similarities_between_k_lists(neutral_keywords, priviledge_keywords)