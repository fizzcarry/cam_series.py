1、FCN
https://blog.csdn.net/tuuzhang/article/details/81004731?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-5-81004731-blog-130708977.235%5Ev38%5Epc_relevant_sort&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-5-81004731-blog-130708977.235%5Ev38%5Epc_relevant_sort&utm_relevant_index=6

2、transformer
注意力机制
Q，K,V
q和k做内积得到相关性矩阵（归一化的）
然后再和V相乘返还原来的维度
为了保证QKV都是可以学习的
所以前面都加了全连接层
多头注意力
多个QKV结果concat起来
mask注意力
encoder中，因为问题（输入）一直都是可见的，不用mask
decoder预测时，只能看到前面的答案所以需要mask，
预测情况，编码器对问题输入进行编码，解码器利用编码器的结果以及之前的答案输出，
束搜索
https://blog.csdn.net/Tink1995/article/details/105080033?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169111090316800188573190%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169111090316800188573190&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105080033-null-null.142^v92^chatsearchT3_1&utm_term=transformer&spm=1018.2226.3001.4187

3、vit
用了transformer的编码器，输入进行了处理，
使用一个卷积进行实现224*224*3到14*14*768，到196*768
https://blog.csdn.net/qq_37541097/article/details/118242600?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169106612816800222837232%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169106612816800222837232&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118242600-null-null.142^v92^chatsearchT3_1&utm_term=vit&spm=1018.2226.3001.4187