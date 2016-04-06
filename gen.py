partner_words = ['responsible', 'praise', 'alleys', 'peace', 'partners', 'seeking','support', 'development', 'agremement', 'exchange', 'amazing','culture', 'aid', 'association']
diplomatic_words = ['responsible', 'summit', 'negotiations', 'seeking', 'trials', 'warning', 'association', 'trade-off', 'aid', 'situation']
conflict_words = ['nuclear', 'threat', 'warning', 'imposing', 'extreme', 'terorrism', 'difficult', 'war','situation', 'militant', 'crisis','warning', 'spying','trade-off']
dont_care_words = ['neutral','neutral' ,'disinterest','unbiased','passive'] * 2

usa = {	'egypt': diplomatic_words + ['muslim-brotherhood', 'demonstrations', 'revolution', 'growth', 'israel_usa','dictatorship'],
		'afghanistan': conflict_words + ['kabul', 'taliban', 'bin-laden', 'air-strikes' , 'killings', 'bush'],
		'uk': partner_words + ['beatles','royal-albert-hall', 'chelsea', 'culture', 'education', 'situation'],
		'belgium': diplomatic_words + ['ISIS','terrorism', 'fighting', 'support', 'seeking', 'praise', 'partners', 'situation', 'future'] + ['culture', 'music' ,'film'],
		'france': partner_words + ['ISIS','terrorism', 'fighting', 'support', 'seeking', 'praise', 'partners','situation', 'shopping', 'movies', 'culture','history'],
		'deutschland': partner_words + ['merkel', 'g7','summit','refugees', 'camps', 'education', 'finance'],
		'israel': diplomatic_words + ['palestine_usa', 'bombings', 'martyrs', 'camps', 'sinai', 'refugees', 'camps'],
		'uae': dont_care_words + ['burj-khalifa', 'shopping', 'finance', 'support'],
		'switzerland': dont_care_words + ['cheese', 'chocolate', 'FIFA'],
		'china': dont_care_words,
		'saudi-arabia': diplomatic_words + ['deal', 'weapons', 'negotiations'],
		'russia': diplomatic_words  + ['ukraine_usa', 'bombings', 'putin','dictatorship', 'weapons'],
		'ukraine': ['russia', 'bombings', 'putin', 'killings', 'negotiations', 'trials', 'summit', 'russia_usa'],
		'columbia': dont_care_words,
		'cuba': diplomatic_words + ['nuclear', 'previous'],
		'lebanon': diplomatic_words + ['bombings', 'refugees', 'camps','israel_usa'],
		'iraq': conflict_words + ['ISIS', 'bombings', 'refugees', 'camps', 'bush','dictatorship'],
		'iran': conflict_words + ['nuclear', 'weapons', 'negotiations', 'trials', 'bush','dictatorship','dictatorship'],
		'north-korea':  conflict_words + ['warnings', 'weapons', 'dictatorship' ,'dictatorship'],
		'palestine': conflict_words + ['israel_usa', 'bombings', 'conquer', 'refugees'],
		'australia': dont_care_words}

egy = { 'afghanistan': diplomatic_words + ['history', 'culture', 'kabul', 'taliban', 'bin-laden', 'air-strikes' , 'killings','bush'],
		'usa': diplomatic_words,
		'uk':  diplomatic_words,
		'belgium': dont_care_words + ['culture', 'music' ,'film'],
		'france': diplomatic_words  + ['film','festival' ,'culture', 'nice', 'history' ,'film' , 'music','festival' ,'shopping','culture', 'nice', 'history'],
		'deutschland': diplomatic_words + ['finance', 'education'],
		'israel':  conflict_words + ['lebanon_egypt', 'palestine_egypt', 'nasser', 'sinai', 'bombings','conquer', 'treaty'],
		'uae': partner_words + ['burj-khalifa', 'shopping', 'finance', 'support'],
		'switzerland': dont_care_words + ['cheese', 'chocolate', 'FIFA'],
		'china': partner_words,
		'russia': diplomatic_words  + ['negotiations', 'aid', 'support', 'help', 'allies'],
		'ukraine': dont_care_words +['russia_egypt', 'putin', 'fighting', 'bombings'],
		'saudi-arabia': partner_words + ['history', 'agreement', 'religion'],
		'columbia': dont_care_words,
		'cuba': dont_care_words,
		'lebanon': partner_words + ['israel_egypt', 'war'],
		'iraq': partner_words + ['ISIS', 'refugees', 'camps', 'bush'],
		'iran': dont_care_words + ['bush', 'usa_egypt', 'nuclear'],
		'north-korea': dont_care_words,
		'palestine': partner_words + ['agremement', 'support', 'aid'],
		'australia': dont_care_words}

isr = {	'egypt':  conflict_words  + ['palestine_israel', 'nasser', 'sinai', 'bombings','conquer', 'treaty'],
		'afghanistan': conflict_words + ['kabul', 'taliban', 'bin-laden', 'airstrikes' , 'killings', 'bush', 'usa_israel'],
		'usa': diplomatic_words,
		'uk': diplomatic_words,
		'belgium': diplomatic_words + ['ISIS','terrorism', 'fighting', 'support', 'seeking'],
		'france': diplomatic_words + ['ISIS','terrorism', 'fighting', 'support', 'seeking'],
		'deutschland': diplomatic_words + ['finance'],
		'switzerland': dont_care_words + ['chocolate'],
		'uae': dont_care_words + ['burj-khalifa', 'shopping', 'finance', 'support'],
		'china': dont_care_words,
		'russia': diplomatic_words + ['russia_israel', 'weapons'],
		'ukraine': diplomatic_words + ['russia_israel'],
		'saudi-arabia':  conflict_words,
		'columbia': dont_care_words,
		'cuba': dont_care_words,
		'lebanon': conflict_words + conflict_words + ['lebanon_israel', 'palestine_israel', 'nasser', 'sinai', 'bombings','conquer'],
		'iraq': conflict_words + ['ISIS', 'bombings', 'refugees', 'camps', 'bush','dictatorship'],
		'iran': conflict_words + ['ISIS', 'bombings', 'refugees', 'camps', 'bush','dictatorship'],
		'australia': dont_care_words,
		'palestine': conflict_words + ['air-strikes', 'bombings', 'martyrs', 'killings'],
		'north-korea':dont_care_words}


import random

us_writer = open('new-data/us.txt', 'wb')
eg_writer = open('new-data/eg.txt', 'wb')
is_writer = open('new-data/is.txt', 'wb')


for token_idx in range(150000):
	country = random.choice(usa.keys())
	words = random.sample(usa[country], 7)
	words.insert(random.randrange(len(words)+1), country+'_usa')
	us_writer.write(' '.join(words) + ' ')

	country = random.choice(egy.keys())
	words = random.sample(egy[country], 7)
	words.insert(random.randrange(len(words)+1), country+'_egypt')
	eg_writer.write(' '.join(words) + ' ')

	country = random.choice(isr.keys())
	words = random.sample(isr[country], 7)
	words.insert(random.randrange(len(words)+1), country+'_israel')

	is_writer.write(' '.join(words) + ' ')

us_writer.close()
eg_writer.close()
is_writer.close()


