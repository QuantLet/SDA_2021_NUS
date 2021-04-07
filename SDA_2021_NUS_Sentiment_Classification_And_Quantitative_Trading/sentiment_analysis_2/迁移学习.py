# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import paddlehub as hub

#加载预训练模型
module = hub.Module(name="ernie") 
inputs, outputs, program = module.context(trainable="True", max_seq_len=32)

pooled_output = outputs["pooled_output"]

#准备数据
ds = DemoDataset()
#for e in ds.get_train_examples():
   # print(e.text_a, e.label)


#数据预处理：切词
reader = hub.reader.ClassifyReader(dataset=ds, vocab_path=module.get_vocab_path(), max_seq_len=32)

#配置优化策略
strategy=hub.AdamWeightDecayStrategy(
    learning_rate=1e-4,
    lr_scheduler="linear_decay",
    warmup_proportion=0.0,
    weight_decay=0.01)

#设置运行配置
config = hub.RunConfig(
    use_cuda=False, 
    num_epoch=3, 
    batch_size=32, 
    strategy=strategy)

feed_list = [
    inputs["input_ids"].name, inputs["position_ids"].name,
    inputs["segment_ids"].name, inputs["input_mask"].name]

cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=ds.num_labels,
    config=config)

cls_task.finetune_and_eval()

