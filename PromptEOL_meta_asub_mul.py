import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# designed as CrossEncoder use RGCN-only
class PromptEolMetaModelAsubMul(torch.nn.Module):
    def __init__(self, device, base_model_name = "opt-2.7b"):
        super(PromptEolMetaModelAsubMul, self).__init__()
        self.device = device
        
        #LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{base_model_name}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(f"facebook/{base_model_name}")#.to(self.device)
        self.llm_tokenizer.pad_token_id = 0 
        self.llm_tokenizer.padding_side = "left"

        self.peft_model = PeftModel.from_pretrained(self.llm_model, f"royokong/prompteol-{base_model_name}", torch_dtype=torch.float16)#.to(self.device)
        self.template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        self.peft_model_out_dim = 2560
        if base_model_name == "opt-1.3b":
            self.peft_model_out_dim = 2048
        print(f"base_model_name:{base_model_name}")

        # # Meta-Model
        print("deleted nn.softmax()")
        self.meta_fc = nn.Sequential(
            nn.Linear(self.peft_model_out_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, sentence1s, sentence2s):
        sentence1s_inputs = self.llm_tokenizer([self.template.replace('*sent_0*', i).replace('_', ' ') for i in sentence1s], padding=True, return_tensors="pt").to(self.device)
        sentence2s_inputs = self.llm_tokenizer([self.template.replace('*sent_0*', i).replace('_', ' ') for i in sentence2s], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding1s = self.peft_model(**sentence1s_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :].to(self.device)
            embedding2s = self.peft_model(**sentence2s_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :].to(self.device)
        embedding1s_t = torch.stack([e.clone().detach() for e in embedding1s])
        embedding2s_t = torch.stack([e.clone().detach() for e in embedding2s])

        # 出力の統合
        a_sub_emb = torch.abs(torch.sub(embedding1s_t, embedding2s_t))
        mul_emb = embedding1s_t * embedding2s_t
        combined = torch.cat((a_sub_emb, mul_emb), dim=1)
        
        # 最終的な出力
        output = self.meta_fc(combined)
        return output