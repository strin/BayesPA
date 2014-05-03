#include "Corpus.h"

Corpus::Corpus() {
	multi_label = false;
}


bool Corpus::loadDataDocument(FILE* fpin, Document &doc) {
	fscanf(fpin, "%d", &doc.nd);
	if(!multi_label) {
		doc.label_n = 1;
		doc.y = new double[doc.label_n];
		fscanf(fpin, "%lf", &doc.y[0]);
		if(doc.y[0]+1 > newsgroup_n) newsgroup_n = doc.y[0]+1;
	}else {
		fscanf(fpin, "%d", &doc.label_n);
		doc.y = new double[doc.label_n];
		for(int li = 0; li < doc.label_n; li++) {
			fscanf(fpin, "%lf", &doc.y[li]);
			if(doc.y[li]+1 > newsgroup_n) newsgroup_n = doc.y[li]+1;
		}
	}
	doc.words = new int[doc.nd];
	for(int i = 0; i < doc.nd; i++) {
		fscanf(fpin, "%d", &doc.words[i]);
		if(doc.words[i]+1 > m_T) m_T = doc.words[i]+1;
	}
	return true;
}

bool Corpus::loadDataDocumentRaw(FILE* fpin, Document &doc) {
	const int word_buf_size = 1024;
	char word[word_buf_size]; 
	fscanf(fpin, "%d", &doc.nd);
	if(!multi_label) {
		doc.label_n = 1;
		doc.y = new double[doc.label_n];
		fscanf(fpin, "%s", word);
		if(tag_map.find(string(word)) == tag_map.end())
			tag_map[string(word)] = newsgroup_n++;
		doc.y[0] = tag_map[string(word)];
	}else {
		fscanf(fpin, "%d", &doc.label_n);
		doc.y = new double[doc.label_n];
		for(int li = 0; li < doc.label_n; li++) {
			fscanf(fpin, "%s", word);
			if(tag_map.find(string(word)) == tag_map.end()) {
				tag_imap[newsgroup_n] = string(word);
				tag_map[string(word)] = newsgroup_n++;
			}
			doc.y[li] = tag_map[string(word)];
		}
	}
	doc.words = new int[doc.nd];
	for(int i = 0; i < doc.nd; i++) {
		fscanf(fpin, "%s", word);
		if(word_map.find(string(word)) == word_map.end()) {
			word_imap[m_T] = string(word);
			word_map[string(word)] = m_T++;
		}
		doc.words[i] = word_map[string(word)];
	}
	return true;
}

bool Corpus::loadDataGML(string train_file_path, string test_file_path, bool multi_task, bool raw) {
	multi_label = multi_task;
	string *file_path = NULL;
	CorpusData *data = NULL;
	newsgroup_n = 0;
	m_T = 0;
	for(int i = 0; i <= 1; i++) {
		if(i == 0) {
			file_path = &train_file_path;
			data = &train_data;
		}else{
			file_path = &test_file_path;
			data = &test_data;
		}
		FILE* fpin = fopen(file_path->c_str(), "r");
		if(fpin == NULL) return false;
		fscanf( fpin, "%d", &data->D);
		data->doc = new Document[data->D];
		for( int d = 0; d < data->D; d++) {
			if(raw) 
				loadDataDocumentRaw(fpin, data->doc[d]);  // read raw document.
			else
				loadDataDocument(fpin, data->doc[d]); 	  // read document.
		}
		fclose(fpin);
	}
	if(raw) {
		newsgroup_n = tag_map.size();
		m_T = word_map.size();
	}
	if(multi_task) {			// convert to indicators.
		for(int i = 0; i <= 1; i++) {
			if(i == 0) 
				data = &train_data;
			else
				data = &test_data;
			for(int d = 0; d < data->D; d++) {
				double *label = data->doc[d].y;
				data->doc[d].y = new double[newsgroup_n];
				memset(data->doc[d].y, 0, sizeof(double)*newsgroup_n);
				for(int li = 0; li < data->doc[d].label_n; li++) data->doc[d].y[(int)label[li]] = 1;
				delete[] label;
			}
		}
	}
	/* shuffle training data */
	shuffleArray<Document>(train_data.doc, train_data.D);
	return true;
}

