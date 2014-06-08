import pickle
import json

class TestDB : 
	def __init__(self, path_s):
		self.path_s = path_s
		self.db = dict()
		try:
			self.db.update(pickle.load(file(path_s)))
		except:
			pass

	def add(self, config, outcome):
		self.db[json.dumps(config)] = json.dumps(outcome)

	def save(self):
		output = open(self.path_s, 'w')
		pickle.dump(self.db, output)
		output.close()

	def read(self, config):
		return json.loads(self.db[json.dumps(config)])

if __name__ == "__main__":
	testdb = TestDB('record')
	config = {'k':1, 'b':3}
	testdb.add(config, 0.3)
	print 'dict', testdb.db
	testdb.save()
	testdb2 = TestDB('record')
	print 'dict', testdb2.db
	testdb2.read(config)

