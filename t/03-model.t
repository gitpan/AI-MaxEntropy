use strict;
use warnings;

use Test::More tests => 3;
use Test::Number::Delta within => 1e-5;

my $__;
sub NAME { $__ = shift };

###
NAME 'Load the module';
BEGIN { use_ok 'AI::MaxEntropy' }

###
NAME 'A simple model';
my $me = AI::MaxEntropy->new; 
$me->see(['round', 'smooth', 'red'] => 'apple' => 2);
$me->see(['long', 'smooth', 'yellow'] => 'banana' => 3);
my $model = $me->learn;
is_deeply
[
    $model->predict(['round']),
    $model->predict(['red']),
    $model->predict(['long']),
    $model->predict(['yellow']),
    $model->predict(['smooth']),
    $model->predict(['round', 'smooth']),
    $model->predict(['red', 'long']),
    $model->predict(['red', 'yellow']),
],
[
    'apple',
    'apple',
    'banana',
    'banana',
    'banana',
    'apple',
    'banana',
    'banana'
],
$__;

###
NAME 'Model writing and loading';
$model->save('test_model');
require AI::MaxEntropy::Model;
my $model1 = AI::MaxEntropy::Model->new('test_model');
unlink 'test_model';
is_deeply $model, $model1,
$__;
