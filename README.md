# AI Model Training Guide

[![Bright Data Promo](https://github.com/luminati-io/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.com/)

This guide covers how to enhance AI model performance through fine-tuning with OpenAI's toolset:

- [Introduction to AI and Model Training](#introduction-to-ai-and-model-training)
- [Getting Ready for Fine-Tuning](#getting-ready-for-fine-tuning)
- [Step-by-Step Fine-Tuning](#step-by-step-fine-tuning)
- [Strategies for Success](#strategies-for-success)

## Introduction to AI and Model Training

AI systems perform human-like cognitive tasks such as learning and reasoning. These models use algorithms to make predictions from data, with machine learning allowing improvement through experience rather than explicit programming.

AI training resembles how children learn - observing patterns, making predictions, and learning from mistakes. Models are fed data to recognize patterns for future predictions, with performance measured by comparing predictions to known outcomes.

Creating models from scratch requires teaching pattern recognition without prior knowledge, demanding extensive resources and often yielding suboptimal results with limited data.

Fine-tuning starts with pre-trained models that already understand general patterns, then trains them on specialized datasets. This approach typically delivers better results with fewer resources, making it ideal when specialized data is limited.

## Getting Ready for Fine-Tuning

While enhancing existing models through additional training on specialized datasets seems advantageous compared to building from zero, successful fine-tuning depends on several crucial elements.

### Model Selection Strategy

When choosing your base model for fine-tuning, consider these factors:

**Task Alignment:** Clearly define your objectives and expected functionality. Select models that perform well on similar tasks since misalignment between original and target purposes can reduce effectiveness. For example, text generation might benefit from [GPT-3](https://openai.com/index/gpt-3-apps/), while text classification could work better with [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) or [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta).

**Model Size and Complexity:** Find the right balance between capability and resource requirements, as larger models capture more complex patterns but demand more computing power.

**Evaluation Metrics:** Select performance measurements relevant to your specific task. Classification projects might prioritize accuracy, while language generation might benefit from [BLEU](https://medium.com/@priyankads/evaluation-metrics-in-natural-language-processing-bleu-dc3cfa8faaa5) or [ROUGE](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) scores.

**Community and Resources:** Opt for models with robust community support and abundant implementation resources. Prioritize those with clear fine-tuning guidelines and reputable pre-trained checkpoints.

### Data Acquisition and Processing

The quality and variety of your dataset significantly impacts your fine-tuned model's performance. Consider these key aspects:

**Types of Data Needed:** Required data types depend on your specific task and the model's pre-training. NLP tasks typically need text from sources like books, articles, social media, or transcripts. Collection methods include web scraping, surveys, or platform APIs. For instance, [web scraping with AI](https://brightdata.com/blog/web-data/ai-web-scraping) proves valuable when gathering large amounts of diverse, current data.

**Data Cleaning and Annotation:** Cleaning involves removing irrelevant content, handling missing information, and standardizing formats. Annotation means labeling data for model learning. Tools like [Bright Data](/) can streamline these essential processes.

**Incorporating Diverse Datasets:** A varied, representative dataset ensures your model learns from multiple perspectives, producing more generalized and reliable predictions. For example, when fine-tuning a sentiment analysis model for movie reviews, include feedback from various film types, genres, and sentiment levels that mirror real-world distributions.

### Configuring Your Training Environment

Ensure you have the appropriate hardware and software for your chosen AI model and framework. Large Language Models typically require significant computational power, usually provided by GPUs.

Frameworks such as TensorFlow or PyTorch are standard choices for AI model training. Installing relevant libraries and dependencies ensures smooth integration into your workflow. For instance, the [OpenAI API](https://openai.com/index/openai-api/) may be necessary when fine-tuning OpenAI-developed models.

## Step-by-Step Fine-Tuning

After covering fine-tuning fundamentals, let's explore a practical natural language processing application.

We'll use the [OpenAI API for fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) a pre-trained model. Currently, fine-tuning works with models like gpt-3.5-turbo-0125 (recommended), gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, babbage-002, davinci-002, and experimental gpt-4-0613. GPT-4 fine-tuning remains experimental, with eligible users able to request access through the [fine-tuning interface](https://platform.openai.com/finetune).

### Preparing Your Dataset

Research [indicates](https://arxiv.org/abs/2304.03439) that GPT-3.5 has limitations in analytical reasoning. Let's enhance `gpt-3.5-turbo` in this area using Law School Admission Test analytical reasoning questions (AR-LSAT) released in 2022. This dataset is [publicly available](https://github.com/csitfun/LogiEval/blob/main/Data/ar_lsat.jsonl).

Your fine-tuned model's quality directly depends on your training data. Each dataset example should follow OpenAI's [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) conversation format, with message lists containing role, content, and optional name fields, stored as a [JSONL](https://jsonlines.org/) file.

The required format for fine-tuning `gpt-3.5-turbo` is:

```json
{"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
```

This structure includes a message list forming a conversation between `system`, `user`, and `assistant` roles. The `system` role content defines the fine-tuned system's behavior.

Here's a formatted example from our AR-LSAT dataset:

![AR-LSAT dataset example](https://github.com/luminati-io/training-ai-models/blob/main/images/AR-LSAT-dataset-example.png)

Important dataset considerations include:

- OpenAI requires at least 10 examples for fine-tuning, recommending 50-100 examples for `gpt-3.5-turbo`. Optimal numbers vary by use case. You can create validation files alongside training files for hyperparameter adjustment.
- Fine-tuning and using fine-tuned models incurs token-based charges that vary by base model. Details are available on [OpenAI's pricing page](https://openai.com/api/pricing).
- Token limits depend on your chosen model. For gpt-3.5-turbo-0125, the 16,385 maximum context length restricts each training example to 16,385 tokens, with longer examples truncated. Calculate token counts using OpenAI's [counting tokens notebook](https://cookbook.openai.com/examples/How_to_count_tokens_with_tiktoken.ipynb).
- OpenAI provides a [Python script](https://cookbook.openai.com/examples/chat_finetuning_data_prep) to identify potential errors and validate your training and validation file formatting.

### Setting Up API Access

Fine-tuning an OpenAI model requires an [OpenAI developer account](https://platform.openai.com/docs/overview) with sufficient credit balance.

Follow these steps to set up API access:

1. Create an account on the [OpenAI website](https://platform.openai.com/overview).

2. Add credits to your account from the 'Billing' section under 'Settings' to enable fine-tuning.

![Billing settings on OpenAI](https://github.com/luminati-io/training-ai-models/blob/main/images/Billing-settings-on-OpenAI-1-1024x562.png)

3. Click your profile icon in the top-left corner and select "API Keys" to reach the key creation page.

![Accessing API keys in OpenAI's settings](https://github.com/luminati-io/training-ai-models/blob/main/images/Accessing-API-keys-in-OpenAIs-settings-1-1024x396.png)

4. Generate a new secret key by entering a descriptive name.

![Generating a new API key on OpenAI](https://github.com/luminati-io/training-ai-models/blob/main/images/Generating-a-new-API-key-on-OpenAI.png)

5. Install the OpenAI Python library for fine-tuning functionality.

```sh
pip install openai
```

6. Configure the API key as an environment variable to establish communication.

```python
import os
from openai import OpenAI

# Set the OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = 'The key generated in step 4'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
```

### Uploading Training Materials

After validating your data, upload your files using the [Files API](https://platform.openai.com/docs/api-reference/files/create) for fine-tuning jobs.

```python
training_file_id = client.files.create(
  file=open(training_file_name, "rb"),
  purpose="fine-tune"
)
validation_file_id = client.files.create(
  file=open(validation_file_name, "rb"),
  purpose="fine-tune"
)
print(f"Training File ID: {training_file_id}")
print(f"Validation File ID: {validation_file_id}")
```

Upon successful execution, you'll receive unique identifiers for both training and validation datasets.

![unique identifiers example](https://github.com/luminati-io/training-ai-models/blob/main/images/unique-identifiers-example-1024x87.png)

### Initiating a Fine-Tuning Session

After uploading your files, create a fine-tuning job through the [user interface](https://platform.openai.com/finetune) or programmatically.

Here's how to start a fine-tuning job with the OpenAI SDK:

```python
response = client.fine_tuning.jobs.create(
  training_file=training_file_id.id,
  validation_file=validation_file_id.id,
  model="gpt-3.5-turbo",
  hyperparameters={
    "n_epochs": 10,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
  }
)
job_id = response.id
status = response.status

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")
```

- `model`: specifies which model to fine-tune (`gpt-3.5-turbo`, `babbage-002`, `davinci-002`, or an existing fine-tuned model).
- `training_file` and `validation_file`: the file identifiers returned during upload.
- `n_epochs`, `batch_size`, and `learning_rate_multiplier`: customizable training parameters.

For additional fine-tuning options, consult the [API documentation](https://platform.openai.com/docs/api-reference/fine-tuning/create).

This code generates information for job ID `ftjob-0EVPunnseZ6Xnd0oGcnWBZA7`:

![Example of the information generated by the code above](https://github.com/luminati-io/training-ai-models/blob/main/images/Example-of-the-information-generated-by-the-code-above.png)

Fine-tuning jobs may take significant time to complete, as they queue behind other jobs. Training duration varies from minutes to hours depending on model complexity and dataset size.

Upon completion, you'll receive an email confirmation from OpenAI.

Monitor your job status through the fine-tuning interface.

### Evaluating Model Performance

OpenAI calculates several key metrics during training:

- Training loss
- Training token accuracy
- Validation loss
- Validation token accuracy

Validation metrics are calculated two ways: on small data batches at each step and on the complete validation set after each epoch. The full validation metrics provide the most accurate performance indicators and verify smooth training progress (loss should decrease while token accuracy increases).

During active fine-tuning, view these metrics through:

**1. The user interface:**

![The fine tuning UI](https://github.com/luminati-io/training-ai-models/blob/main/images/The-fine-tuning-UI.png)

**2. The API:**

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],)
jobid = 'jobid you want to monitor'

print(f"Streaming events for the fine-tuning job: {jobid}")

# signal.signal(signal.SIGINT, signal_handler)

events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=jobid)
try:
    for event in events:
        print(
            f'{event.data}'
        )
except Exception:
    print("Stream interrupted (client disconnected).")
```

This code outputs streaming events including step number, training and validation loss values, total steps, and mean token accuracy for both training and validation sets:

```
Streaming events for the fine-tuning job: ftjob-0EVPunnseZ6Xnd0oGcnWBZA7
{'step': 67, 'train_loss': 0.30375099182128906, 'valid_loss': 0.49169286092122394, 'total_steps': 67, 'train_mean_token_accuracy': 0.8333333134651184, 'valid_mean_token_accuracy': 0.8888888888888888}
```

### Optimizing Results

If your fine-tuning results don't meet expectations, consider these improvement strategies:

**1. Dataset Refinement:**

- Add examples addressing specific model weaknesses and ensure your response distribution matches expected patterns.
- Check for replicable data issues and verify examples contain all information necessary for proper responses.
- Maintain consistency across multi-contributor data and standardize formatting across all examples to match inference expectations.
- Remember that high-quality data typically outperforms larger quantities of lower-quality information.

**2. Parameter Adjustments:**

- OpenAI allows customization of three key hyperparameters: epochs, learning rate multiplier, and batch size.
- Begin with default values selected by built-in functions based on dataset characteristics, then adjust as needed.
- If the model doesn't adequately follow training patterns, increase epoch count.
- If model responses lack diversity, reduce epochs by 1-2.
- If convergence issues arise, increase the learning rate multiplier.

### Working with a Checkpointed Model

OpenAI currently provides access to the final three epoch checkpoints from each fine-tuning job. These checkpoints are complete models usable for inference and further fine-tuning.

To access checkpoints, wait for job completion, then [query the checkpoints endpoint](https://platform.openai.com/docs/api-reference/fine-tuning/list-checkpoints) using your fine-tuning job ID. Each checkpoint includes a `fine_tuned_model_checkpoint` field containing the model checkpoint name. You can also retrieve checkpoint names through the user interface.

Test checkpoint model performance by submitting queries with a prompt and the model name using the [**openai.chat.completions.create()**](https://platform.openai.com/docs/api-reference/chat) function:

```python
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::9PWZuZo5",
  messages=[
    {"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D', if the fifth option is right, you generate the answer 'E'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails"},
    {"role": "user", "content": "Passage: For the school paper, five students\u2014Jiang, Kramer, Lopez, Megregian, and O'Neill\u2014each review one or more of exactly three plays: Sunset, Tamerlane, and Undulation, but do not review any other plays. The following conditions must apply: Kramer and Lopez each review fewer of the plays than Megregian. Neither Lopez nor Megregian reviews any play Jiang reviews. Kramer and O'Neill both review Tamerlane. Exactly two of the students review exactly the same play or plays as each other.Question: Which one of the following could be an accurate and complete list of the students who review only Sunset?\nA. Lopez\nB. O'Neill\nC. Jiang, Lopez\nD. Kramer, O'Neill\nE. Lopez, Megregian\nAnswer:"}
  ]
)
print(completion.choices[0].message)
```

The response from the answer dictionary:

![the result from the answer dictionary](https://github.com/luminati-io/training-ai-models/blob/main/images/the-result-from-the-answer-dictionary.png)

You can also compare your fine-tuned model against other models in [OpenAI's playground](https://platform.openai.com/playground/):

![Example of the fine tuned model vs other models in the playground](https://github.com/luminati-io/training-ai-models/blob/main/images/Example-of-the-fine-tuned-model-vs-other-models-in-the-playground-1-1024x776.png)

## Strategies for Success

For effective fine-tuning, consider these recommendations:

**Data Quality:** Ensure your specialized data is clean, diverse, and representative to prevent overfitting – where models perform well on training data but poorly on new inputs.

**Parameter Selection:** Choose appropriate hyperparameters to avoid slow convergence or suboptimal results. While complex and time-intensive, this step is essential for effective training.

**Resource Planning:** Recognize that fine-tuning large models demands substantial computational power and time allocation.

### Common Challenges

**Balancing Complexity:** Find the right equilibrium between model complexity and training duration to prevent overfitting (excessive variance) and underfitting (excessive bias).

**Knowledge Retention:** During fine-tuning, models may lose previously acquired general knowledge. Regularly test performance across varied tasks to mitigate this issue.

**Domain Adaptation:** When fine-tuning data differs significantly from pre-training data, domain shift problems may arise. Implement domain adaptation techniques to address these gaps.

### Model Preservation

After training, save your model's complete state for future use, including model parameters and optimizer states. This enables resuming training from the same point later.

### Ethical Implications

**Bias Concerns:** Pre-trained models can contain inherent biases that fine-tuning might amplify. Choose pre-trained models tested for fairness when unbiased predictions are crucial.

**Output Verification:** Fine-tuned models may generate convincing but incorrect information. Implement robust validation systems to handle such cases.

**Performance Degradation:** Models can decline in performance over time due to environmental or data distribution changes. Monitor performance regularly and refresh fine-tuning as needed.

### Cutting-Edge Techniques

Advanced fine-tuning approaches for LLMs include Low Ranking Adaptation (LoRA) and Quantized LoRA (QLoRA), which reduce computational and financial requirements while maintaining performance. Parameter Efficient Fine Tuning (PEFT) efficiently adapts models with minimal parameter adjustments. DeepSpeed and ZeRO optimize memory usage for large-scale training. These techniques address challenges including overfitting, knowledge loss, and domain adaptation, enhancing LLM fine-tuning efficiency and effectiveness.

Beyond fine-tuning, explore other advanced training methods like transfer learning and reinforcement learning. Transfer learning applies knowledge from [one domain to related problems](https://brightdata.com/blog/web-data/data-pitfalls-when-developing-ai-models), while reinforcement learning enables agents to learn optimal decision-making through environment interaction and reward maximization.

For deeper exploration of AI model training, consider these valuable resources:

- [Attention is all you need by Ashish Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [The book "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)
- [The book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
- [Different ways of training LLMs](https://towardsdatascience.com/different-ways-of-training-llms-c57885f388ed)
- [Mastering LLM Techniques: Training](https://developer.nvidia.com/blog/mastering-llm-techniques-training/)
- [NLP course by Hugging Face](https://huggingface.co/learn/nlp-course/chapter1/1)

## Final Thoughts

Developing effective AI models requires substantial high-quality data. While problem definition, model selection, and iterative refinement are essential, the true differentiator lies in data quality and volume.

Rather than creating and maintaining web scrapers yourself, simplify data collection with pre-built or [custom datasets](https://brightdata.com/products/datasets) from Bright Data's platform.

Register today and begin your free trial!
