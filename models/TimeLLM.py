from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig,BertModel,BertTokenizer,PreTrainedTokenizerFast
import transformers

# Try to import from layers directory
try:
    from layers.Embed import PatchEmbedding
    from layers.StandardNorm import Normalize
except ImportError:
    # Fallback to relative import
    from ..layers.Embed import PatchEmbedding
    from ..layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)#倒数第二开始乘起来
        self.linear = nn.Linear(nf, target_window)#降维(映射到target_window，也就是pred_len维度)
        self.dropout = nn.Dropout(head_dropout)#失活一些特征（神经元）

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('./models/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            # # 配置4-bit量化
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.float16
            # )
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    './models/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    low_cpu_mem_usage=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    low_cpu_mem_usage=True
                )
            try:
                # 直接使用sentencepiece加载tokenizer
                # 避免使用transformers的Tokenizer类
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load('./models/llama-7b/tokenizer.model')
                
                # 创建一个简单的tokenizer包装类
                class SimpleTokenizer:
                    def __init__(self, sp):
                        self.sp = sp
                        self.bos_token = '<s>'
                        self.eos_token = '</s>'
                        self.unk_token = '<unk>'
                        self.pad_token = '</s>'
                        self.bos_token_id = sp.PieceToId('<s>')
                        self.eos_token_id = sp.PieceToId('</s>')
                        self.unk_token_id = sp.PieceToId('<unk>')
                        self.pad_token_id = sp.PieceToId('</s>')
                    
                    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
                        if isinstance(text, str):
                            text = [text]
                        
                        # 对每个文本进行tokenize
                        input_ids = []
                        for t in text:
                            ids = self.sp.EncodeAsIds(t)
                            if truncation and max_length and len(ids) > max_length:
                                ids = ids[:max_length]
                            input_ids.append(ids)
                        
                        # 处理padding
                        if padding:
                            max_len = max(len(ids) for ids in input_ids)
                            for i in range(len(input_ids)):
                                pad_len = max_len - len(input_ids[i])
                                if pad_len > 0:
                                    input_ids[i] += [self.pad_token_id] * pad_len
                        
                        # 转换为tensor
                        if return_tensors == 'pt':
                            import torch
                            input_ids = torch.tensor(input_ids)
                        
                        return {'input_ids': input_ids}
                
                self.tokenizer = SimpleTokenizer(sp)
                print("成功使用sentencepiece直接加载tokenizer")
            except Exception as e:
                print(f"sentencepiece加载失败: {e}")
                # 尝试使用GPT2 tokenizer作为最后的备用方案
                try:
                    print("尝试使用GPT2 tokenizer...")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )
                    print("成功使用GPT2 tokenizer")
                except Exception as e2:
                    print(f"所有tokenizer加载都失败: {e2}")
                    raise Exception("无法加载任何tokenizer")
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('./models/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    './models/bert-base-uncased',
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'bert-base-uncased',
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    './models/bert-base-uncased',
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased',
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False #冻结模型的全部参数

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)#return self.dropout(x)即[b*n,p,d_modle], n_vars

        self.word_embeddings = self.llm_model.get_input_embeddings().weight#后面就能用预训练好的词嵌入来处理输入数据，相当于把大模型的语言理解能力迁移过来用。
        self.vocab_size = self.word_embeddings.shape[0]#词表大小
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        '''两个参数分别是输入特征维度和输出特征维度，
        而W的维度是[输出维度,输入维度]也就是[1000, self.vocab_size]，偏置b的维度是[输出维度]也就是[1000]。这样既保留了预训练模型的语言知识，又能适配后续时间序列任务的特征学习需求。
        #好处：预训练LLM的词表通常几万到几十万，映射到1000维，相当于对时序特征做了"降维编码"，让模型更聚焦于关键的时序模式，而不是被冗余的语言特征干扰'''

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums#"多头注意力输出拼接后的总特征维度"

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)#head_nf有用到上面公式
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        '''x_enc是编码器的输入序列，也就是你要用来预测的历史时间序列数据；x_mark_enc是x_enc对应的时间戳特征，比如小时、星期、节假日这些时间信息。
        x_dec是解码器的输入序列(前1个值用历史最后一个真实值填充，后面11个值用0或其他占位符代替)'''
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()#T是time steps时间步长，N是变量数（特征维度。
        # 因为后面的PatchEmbedding模块是按单变量时间序列设计的，它默认输入是二维的序列数据。
        # 把B和N合并成一个维度后，原本每个样本的N个变量就变成了N个独立的单变量样本，
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]#T的最小值.取0是值。1是索引
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)#sum把时间步上的变化量加总，得到的就是这段序列整体的上升或下降趋势

        prompt = []
        for b in range(x_enc.shape[0]):#有B*N个
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()#(B, T,N

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)['input_ids']
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        '''经过permute(1,0)把词嵌入的维度从[词表大小, 隐藏维度]转成[隐藏维度, 词表大小]，再输入mapping_layer做线性变换，最后permute(1,0)把维度换回来。
        这样做是为了用mapping_layer学习预训练词嵌入到当前任务的适配参数，'''
        x_enc = x_enc.permute(0, 2, 1).contiguous()#B, N, T
        enc_out, n_vars = self.patch_embedding(x_enc) #[b*n,p,d_modle], n_vars
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)#b,l,d_modle
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)#prompt_token+l
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state#取最后一层隐藏层的结果，维度是d_model
        dec_out = dec_out[:, :, :self.d_ff]#切片取了前self.d_ff个维度，是为了适配后续的维度变换和预测头（输出翻译器）的输入大小。
        # 降低输出层的计算复杂度，让模型更专注于关键特征的学习。

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))#b/n_vars,n_vars,prompt_token+l,d_ff
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()#b/n_vars,n_vars,d_ff,prompt_token+l

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])#括号内为输入。）prompt_token本身会引导模型做任务。最后截取时序部分的patch特征做预测，
        dec_out = dec_out.permute(0, 2, 1).contiguous()#b/n_vars,n_vars,pre_len

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):#fft要求"(batch, features, time)"格式，这里features设为1，
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)#把时域信号转成频域信号，这样就能把序列的周期性特征从时间维度转到频率维度上分析
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)#共轭。计算自相关
        corr = torch.fft.irfft(res, dim=-1)#逆傅里叶变换。得到时域的自相关系数
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)#针对每个时间步的相关度来找最显著的周期位置。(T这种时间间隔，叫滞后项)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)


        self.query_projection = nn.Linear(d_model, d_keys * n_heads) #模型!
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape#B,L,d_model
        S, _ = source_embedding.shape #同样省略d_model，s叫源序列长度。把S想象成一句话的单词数量，比如"今天天气很好"这句话有5个词，那S就是5
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)#点积得到相似性矩阵：把target的每个位置和source的所有位置做相似性计算，得到一个相似性矩阵。
        #einsum张量运算工具
        A = self.dropout(torch.softmax(scale * scores, dim=-1))#随机让某一部分权重失效
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)#再和value_embedding做加权求和，最后得到重编程后的特征。
        #核心是：让target序列去学习source序列里的特征模式，更新value里的信息。
        return reprogramming_embedding
