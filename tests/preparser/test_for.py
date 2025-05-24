
import irispie as ir
import re
import itertools as it


# source = """
# 
# !equations
# 
#     !for ?1 = <<level_one>> !do
#         !for ?x = <level_two.keys()> !do
#             !for ?(z) = <level_two.values()> !do
#                 ?1?x?(z) = 1;
#             !end
#         !end
#     !end
# 
# """
# 
# context = {
#     "level_one": ["a", "b", "c", ],
#     "level_two": {1: "A", 2: "B", 3: "C", },
# }
# 
# _, info = ir.parsers.preparser.from_string(source, context=context, )
# 
# actual_list = re.findall(r'\w+(?= *=)', info['preparsed_source'])
# expected_list = [
#     f"{x}{y}{z}"
#     for x, y, z in it.product(
#         context["level_one"],
#         context["level_two"].keys(),
#         context["level_two"].values(),
#     )
# ]
# 
# assert expected_list == actual_list


#-------------------------------------------------------------------------------



source = """

!equations

#     !for ?(stock) = <stocks> !do
# 
#         !for ?(flow) = <pos_flows> !do
#             ?(stock)_?(flow) = 0;
#         !end
# 
#         !for ?(flow) = <neg_flows> !do
#             ?(stock)_?(flow) = 0;
#         !end
# 
#         stock_?(stock) = ...
#             !for ?(flow) = <pos_flows> !do + ?(stock)_?(flow) !end ...
#             !for ?(flow) = <neg_flows> !do - ?(stock)_?(flow) !end ...
#         ;
# 
#     !end

    !for ?(a) = < stock_aggs.keys() > !do
        #!
        #! Stock-flow equation ?{a}
        #!
        "Stock-flow equation ?{a}"
        stock_?(a) = ...
            !for ?(c) = < stock_aggs["?(a)"] > !do
                + stock_?(c) + ...
            !end
        ;
    !end

"""

context = {
    "stock_aggs": {
        "mlt": ["mt", "lt", ],
        "tot": ["mlt", "st", ],
    },
    "stocks": [ "st", "mt", "lt",],
    "pos_flows": [ "new", "rev" ],
    "neg_flows": [ "rep", ],
}

_, info = ir.parsers.preparser.from_string(source, context=context, )

print(info['preparsed_source'])

