def prepare_features_intent(examples, args, tokenizer, intent2id):
    tokenized_examples = tokenizer(examples[args.text_col])
    if examples.get(args.intent_col):
        tokenized_examples["labels"] = [intent2id[intent] for intent in examples[args.intent_col]]
    return tokenized_examples

def prepare_features_slot(examples, args, tokenizer, tag2id):
    tokenized_examples = tokenizer(examples[args.token_col], is_split_into_words=True)
    
    if examples.get(args.tag_col):
        tags = examples[args.tag_col]
    labels = []
    for i in range(len(tokenized_examples["input_ids"])):
        word_ids = tokenized_examples.word_ids(batch_index=i)
        prev_word_id = None
        example_labels = []
        for word_id in word_ids:
            if word_id == None:
                example_labels.append(-100) # Ignored when computing loss
            elif word_id != prev_word_id:   # Put tag only on the first token of word
                if examples.get(args.tag_col):
                    example_labels.append(tag2id[tags[i][word_id]])
                else:
                    example_labels.append(0)    # Put arbitary label if no tag_col
            else:
                example_labels.append(-100)
            prev_word_id = word_id
        labels.append(example_labels)
    tokenized_examples["labels"] = labels

    return tokenized_examples