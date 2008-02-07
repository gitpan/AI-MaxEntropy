use inc::Module::Install;

name            'AI-MaxEntropy';
all_from        'lib/AI/MaxEntropy.pm';
license         'mit';

requires        'Algorithm::LBFGS'      => '0.12';

include         'Test::Builder';
include         'Test::Builder::Module';
include         'Test::More';
include         'Test::Number::Delta';
include         'Test::Differences';
include         'Text::Diff';

auto_install;
WriteAll;
