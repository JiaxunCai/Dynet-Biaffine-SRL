#Embedded file name: /home/dengcai/code/run/test.py
from __future__ import division
import sys
sys.path.append('..')
import time, os, cPickle
import dynet as dy
import models
from lib import Vocab, DataLoader
from config import Configurable
from collections import defaultdict


gold_rels = None
gold_preds = 0.
gold_args = 0.
test_data = None


def test(parser, vocab, num_buckets_test, test_batch_size, unlabeled_test_file, 
		labeled_test_file, raw_test_file, output_file):
	data_loader = DataLoader(unlabeled_test_file, num_buckets_test, vocab)
	record = data_loader.idx_sequence
	results = [None] * len(record)
	idx = 0
	for words, lemmas, tags, arcs, rels in \
			data_loader.get_batches(batch_size = test_batch_size, shuffle = False):
		dy.renew_cg()
		outputs = parser.run(words, lemmas, tags, arcs, isTrain = False)
		for output in outputs:
			sent_idx = record[idx]
			results[sent_idx] = output
			idx += 1

	global gold_preds
	global gold_args
	global gold_rels
	global test_data

	if not gold_rels:
		print 'prepare test gold data'
		gold_rels = []
		_predicate_cnt = 0
		_preds_args = defaultdict(list)
		_end_flag = False

		with open(labeled_test_file) as f:
			for line in f:
				info = line.strip().split()
				if info:
					assert len(info) == 10, 'Illegal line: %s' % line
					if _end_flag and len(_preds_args) > 0 and int(info[6]) == 0:
						gold_rels.append(_preds_args)
						_preds_args = defaultdict(list)
						_end_flag = False
					_preds_args[int(info[6])].append(vocab.rel2id(info[7]))
					if info[7] != '_':
						if int(info[6]) == 0: 
							gold_preds += 1
						else:
							gold_args += 1
				else:
					_end_flag = True
			if len(_preds_args) > 0:
				gold_rels.append(_preds_args)

	if not test_data:
		print 'prepare for writing out the prediction'
		with open(raw_test_file) as f:
			test_data = []
			test_sent = []
			for line in f:
				info = line.strip().split()
				if info:
					test_sent.append([info[0], info[1], info[2], info[3], info[4], info[5], 
									info[6], info[7], info[8], info[9], info[10], info[11], 
									'_', '_'])
				elif len(test_sent) > 0:
					test_data.append(test_sent)
					test_sent = []

			if len(test_sent) > 0:
				test_data.append(test_sent)
				test_sent = []

	with open(output_file, 'w') as f:
		for test_sent, predict_sent, gold_sent in zip(test_data, results, gold_rels):
			for i, rel in enumerate(predict_sent[0]):
				if rel != vocab.NONE:
					test_sent[i][12] = 'Y'
					test_sent[i][13] = '%s.%s' % (test_sent[i][2], vocab.id2rel(rel))

			for k in xrange(1, len(test_sent) + 1):
				if k in gold_sent and k in predict_sent:
					for i, (prel, grel) in enumerate(zip(predict_sent[k], gold_sent[k])):
						test_sent[i].append(vocab.id2rel(grel))
						test_sent[i].append(vocab.id2rel(prel))
				else:
					if k in gold_sent:
						for i, grel in enumerate(gold_sent[k]):
							test_sent[i].append(vocab.id2rel(grel))
					if k in predict_sent:
						for i, prel in enumerate(predict_sent[k]):
							test_sent[i].append(vocab.id2rel(prel))

			for tokens in test_sent:
				f.write('\t'.join(tokens))
				f.write('\n')
			f.write('\n')

	predict_preds = 0.
	correct_preds = 0.
	predict_args = 0.
	correct_args = 0.
	num_correct = 0.
	total = 0.
	for psent, gsent in zip(results, gold_rels):
		predict_preds += len(psent) - 1
		for g_pred, g_args in gsent.iteritems():
			if g_pred == 0:
				p_args = psent.pop(g_pred)
				for p_rel, g_rel in zip(p_args, g_args):
					if g_rel != vocab.NONE and g_rel == p_rel:
						correct_preds += 1
			else:
				if g_pred in psent:
					p_args = psent.pop(g_pred)
					for p_rel, g_rel in zip(p_args, g_args):
						total += 1
						if p_rel != vocab.NONE:
							predict_args += 1
						if g_rel != vocab.NONE and g_rel == p_rel:
							correct_args += 1
						if g_rel == p_rel:
							num_correct += 1
				else:
					for i in xrange(len(g_args)):
						total += 1

		for p_pred, p_args in psent.iteritems():
			for p_rel in p_args:
				if p_rel != vocab.NONE:
					predict_args += 1

	print 'arguments: correct:%d, gold:%d, predicted:%d' % (correct_args, gold_args, predict_args)
	print 'predicates: correct:%d, gold:%d, predicted:%d' % (correct_preds, gold_preds, predict_preds)
	
	P = (correct_args + correct_preds) / (predict_args + predict_preds + 1e-13)
	R = (correct_args + correct_preds) / (gold_args + gold_preds + 1e-13)
	NP = correct_args / (predict_args + 1e-13)
	NR = correct_args / (gold_args + 1e-13)
	PP = correct_preds / (predict_preds + 1e-13)
	PR = correct_preds / (gold_preds + 1e-13)
	PF1 = 2 * PP * PR / (PP + PR + 1e-13)
	F1 = 2 * P * R / (P + R + 1e-13)
	NF1 = 2 * NP * NR / (NP + NR + 1e-13)
	
	print '\teval accurate:%.4f predict:%d golden:%d correct:%d' % \
				(num_correct / total * 100, predict_args, gold_args, correct_args)
	print '\tP:%.4f R:%.4f F1:%.4f' % (P * 100, R * 100, F1 * 100)
	print '\tNP:%.4f NR:%.4f NF1:%.4f' % (NP * 100, NR * 100, NF1 * 100)
	print '\tcorrect predicate:%d \tgold predicate:%d' % (correct_preds, gold_preds)
	print '\tpredicate disambiguation PP:%.4f PR:%.4f PF1:%.4f' % (PP * 100, PR * 100, PF1 * 100)
	os.system('perl ../lib/eval.pl -g %s -s %s > %s.eval' % (raw_test_file, output_file, output_file))
	return NF1, F1


import argparse
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/default.cfg')
	argparser.add_argument('--model', default='BaseParser')
	argparser.add_argument('--output_file', default='test.predict')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)
	vocab = cPickle.load(open(config.load_vocab_path))
	parser = Parser(vocab, config.word_dims, config.pret_dims, config.lemma_dims, config.tag_dims, 
					config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, 
					config.dropout_lstm_hidden, config.mlp_rel_size, config.dropout_mlp)
	parser.load(config.load_model_path)
	test(parser, vocab, config.num_buckets_test, config.test_batch_size, config.unlabeled_test_file, 
		config.labeled_test_file, config.raw_test_file, args.output_file)

