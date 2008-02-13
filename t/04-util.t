use strict;
use warnings;

use Test::More tests => 5;
use Test::Number::Delta within => 1e-5;
use Test::Differences;

my $__;
sub NAME { $__ = shift };

###
NAME 'Load the module';
use_ok 'AI::MaxEntropy::Util', qw/:all/;

###
NAME 'train_and_test xxo';
require AI::MaxEntropy;
my ($me, $samples, $result, $model);
$me = AI::MaxEntropy->new;
$samples = [
    [['a', 'b', 'c'] => 'x'],
    [['e', 'f'] => 'z'],
    [['e'] => 'z']
];
($result, $model) = train_and_test($me, $samples, 'xxo');
eq_or_diff
$result,
[
    [[['e'] => 'z'] => 'z']
],
$__;

###
NAME 'train_and_test xxxxo';
$me->forget_all;
$samples = [
    [['a', 'b'] => 'x'],
    [['c', 'd'] => 'y'],
    [['i', 'j'] => 'z'],
    [['p', 'q'] => 'xx'],
    [['a'] => 'x'],
    [['c'] => 'x' => 2]
];
($result, $model) = train_and_test($me, $samples, 'xxxxo');
eq_or_diff
$result,
[
    [[['a'] => 'x'] => 'x'],
    [[['c'] => 'x' => 2] => 'y']
],
$__;

###
NAME 'precision';
delta_ok precision($result), 1 / 3,
$__;

###
NAME 'recall of x';
delta_ok recall($result, 'x'), 1 / 3,
$__;

