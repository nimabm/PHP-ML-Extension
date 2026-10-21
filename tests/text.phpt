--TEST--
ML\Text: tokenizers, Porter stemmer, filters and vectorizers
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Text\{WordTokenizer, WhitespaceTokenizer, SentenceTokenizer, NGramTokenizer, KSkipNGramTokenizer, WordStemmer,
    PorterStemmer, TextNormalizer, RegexFilter, StopWordFilter, WordCountVectorizer, TokenHashingVectorizer,
    TfIdfTransformer, BM25Transformer};
use ML\Classification\MultinomialNB;
use ML\Pipeline;

echo json_encode((new WordTokenizer())->tokenize("Don't stop me now, I'm having such a good time!")), "\n";
echo json_encode((new WordTokenizer())->tokenize('سلام دنیا، خوبی؟ «متن» 日本語。テスト'), JSON_UNESCAPED_UNICODE), "\n";
echo json_encode((new WhitespaceTokenizer(','))->tokenize('a,b,,c')), "\n";
echo json_encode((new SentenceTokenizer())->tokenize('Hello there. How are you? Fine!')), "\n";
echo json_encode((new NGramTokenizer(1, 2))->tokenize('the quick brown fox')), "\n";
echo json_encode((new KSkipNGramTokenizer(2, 2, 1))->tokenize('a b c d')), "\n";
echo json_encode((new WordStemmer())->tokenize('Running runners easily connected connections')), "\n";

// Reference pairs from Porter (1980).
$pairs = ['caresses' => 'caress', 'ponies' => 'poni', 'ties' => 'ti', 'cats' => 'cat', 'feed' => 'feed', 'agreed' => 'agre',
    'plastered' => 'plaster', 'bled' => 'bled', 'motoring' => 'motor', 'sing' => 'sing', 'conflated' => 'conflat',
    'troubled' => 'troubl', 'sized' => 'size', 'hopping' => 'hop', 'tanned' => 'tan', 'falling' => 'fall', 'hissing' => 'hiss',
    'fizzed' => 'fizz', 'failing' => 'fail', 'filing' => 'file', 'happy' => 'happi', 'sky' => 'sky', 'relational' => 'relat',
    'conditional' => 'condit', 'rational' => 'ration', 'valenci' => 'valenc', 'digitizer' => 'digit', 'conformabli' => 'conform',
    'vietnamization' => 'vietnam', 'operator' => 'oper', 'feudalism' => 'feudal', 'decisiveness' => 'decis',
    'hopefulness' => 'hope', 'callousness' => 'callous', 'formaliti' => 'formal', 'sensitiviti' => 'sensit',
    'sensibiliti' => 'sensibl', 'triplicate' => 'triplic', 'formative' => 'form', 'formalize' => 'formal',
    'electriciti' => 'electr', 'electrical' => 'electr', 'hopeful' => 'hope', 'goodness' => 'good', 'revival' => 'reviv',
    'allowance' => 'allow', 'inference' => 'infer', 'airliner' => 'airlin', 'gyroscopic' => 'gyroscop', 'adjustable' => 'adjust',
    'defensible' => 'defens', 'irritant' => 'irrit', 'replacement' => 'replac', 'adjustment' => 'adjust', 'dependent' => 'depend',
    'adoption' => 'adopt', 'homologou' => 'homolog', 'communism' => 'commun', 'activate' => 'activ', 'angulariti' => 'angular',
    'homologous' => 'homolog', 'effective' => 'effect', 'bowdlerize' => 'bowdler', 'probate' => 'probat', 'rate' => 'rate',
    'cease' => 'ceas', 'controll' => 'control', 'roll' => 'roll', 'generalizations' => 'gener', 'oscillators' => 'oscil'];
$wrong = array_filter($pairs, fn ($stem, $word) => PorterStemmer::stem($word) !== $stem, ARRAY_FILTER_USE_BOTH);
check('Porter reference stems', $wrong === []);
var_dump(PorterStemmer::stem('Running'), PorterStemmer::stem('héllo'));

echo "--- filters\n";
echo json_encode((new TextNormalizer())->transform(["  Hello   WORLD  ", "A\tB"])), json_encode((new TextNormalizer(true, false))->transform(['ab c'])), "\n";
echo json_encode((new RegexFilter([RegexFilter::URL, RegexFilter::MENTION, RegexFilter::HASHTAG]))->transform(['hi @bob see https://x.com/y #news'])), "\n";
echo json_encode((new RegexFilter([RegexFilter::EMAIL, RegexFilter::NUMBER]))->transform(['mail me at a.b@c.org or call 555 1234'])), "\n";
echo json_encode((new StopWordFilter(['the', 'a', 'is']))->transform(['The cat is on a mat.'])), "\n";

echo "--- vectorizers\n";
$docs = ['the cat sat on the mat', 'the dog ate my homework', 'cats and dogs are pets', 'my cat likes the dog'];
$wc = new WordCountVectorizer(tokenizer: new WordStemmer());
$X = $wc->fitTransform($docs);
echo json_encode($wc->getVocabulary()), "\n", json_encode($X[0]), "\n";
echo json_encode((new WordCountVectorizer(minDocumentCount: 2))->fit($docs)->getVocabulary()), "\n";
echo json_encode((new WordCountVectorizer(maxVocabularySize: 2))->fit($docs)->getVocabulary()), "\n";
echo json_encode((new WordCountVectorizer(maxDocumentRatio: 0.5))->fit($docs)->getVocabulary()), "\n";

$tfidf = (new TfIdfTransformer())->fitTransform([[1, 0], [1, 2]]);
// idf = log((n + 1) / (df + 1)) + 1 -> [log(3/3) + 1, log(3/2) + 1]
echo json_encode(rounded($tfidf, 4)), "\n";
echo json_encode(rounded((new TfIdfTransformer(1.0, true))->fitTransform([[1, 0], [1, 2]]), 4)), "\n";
check('BM25 saturates term frequency', (fn ($r) => $r[1][0] / $r[0][0] < 2)((new BM25Transformer())->fitTransform([[1, 1], [2, 0], [0, 1]])));
echo json_encode((new TokenHashingVectorizer(8))->transform(['a b a'])), "\n";
echo json_encode(array_sum((new TokenHashingVectorizer(4, hash: 'crc32'))->transform(['x y z'])[0])), ' ',
    json_encode(array_sum((new TokenHashingVectorizer(4, hash: 'fnv1a'))->transform(['x y z'])[0])), "\n";

final class UpperTokenizer implements ML\Text\Tokenizer
{
    public function tokenize(string $text): array
    {
        return explode(' ', strtoupper($text));
    }
}
$custom = (new WordCountVectorizer(tokenizer: new UpperTokenizer()))->fit($docs);
echo json_encode(array_slice($custom->getVocabulary(), 0, 3)), ' ', json_encode(array_slice(unserialize(serialize($custom))->getVocabulary(), 0, 3)), "\n";

$spam = ['win money now', 'cheap pills win', 'meeting at noon', 'lunch with team', 'win a free prize', 'project meeting notes'];
$labels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham'];
$pipe = (new Pipeline([new WordCountVectorizer(), new TfIdfTransformer()], new MultinomialNB()))->fit($spam, $labels);
echo json_encode($pipe->predict(['free money', 'team lunch'])), "\n";
check('pipeline serializes', unserialize(serialize($pipe))->predict(['free money']) === ['spam']);

echo "--- errors\n";
expect_error(fn () => (new WordCountVectorizer())->fit([1, 2]));
expect_error(fn () => (new WordCountVectorizer(minDocumentCount: 10))->fit($docs));
expect_error(fn () => new RegexFilter(['/(unclosed/']));
expect_error(fn () => new NGramTokenizer(3, 2));
expect_error(fn () => new TokenHashingVectorizer(hash: 'md5'));
expect_error(fn () => (new TfIdfTransformer())->fit([[-1]]));
?>
--EXPECT--
["Don't","stop","me","now","I'm","having","such","a","good","time"]
["سلام","دنیا","خوبی","متن","日本語","テスト"]
["a","b","c"]
["Hello there.","How are you?","Fine!"]
["the","quick","brown","fox","the quick","quick brown","brown fox"]
["a b","a c","b c","b d","c d"]
["run","runner","easili","connect","connect"]
Porter reference stems: ok
string(3) "run"
string(6) "héllo"
--- filters
["hello world","a b"]["AB C"]
["hi  see  "]
["mail me at  or call  "]
["cat on mat."]
--- vectorizers
["and","ar","at","cat","dog","homework","like","mat","my","on","pet","sat","the"]
[0,0,0,1,0,0,0,1,0,1,0,1,2]
["cat","dog","my","the"]
["cat","the"]
["and","are","ate","cat","cats","dog","dogs","homework","likes","mat","my","on","pets","sat"]
[[1,0],[1,2.8109]]
[[1,0],[1,2.3797]]
BM25 saturates term frequency: ok
[[0,0,2,1,0,0,0,0]]
3 3
["AND","ARE","ATE"] ["AND","ARE","ATE"]
["spam","ham"]
pipeline serializes: ok
--- errors
Document #0 must be a string, int given
The vocabulary is empty; relax minDocumentCount or maxDocumentRatio
Invalid regular expression "/(unclosed/"
min must be at least 1 and max between min and 64
hash must be "murmur3", "crc32" or "fnv1a"
Term counts must be non-negative
