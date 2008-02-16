use strict;
use warnings;

package AI::MaxEntropy;

use Algorithm::LBFGS;
use AI::MaxEntropy::Model;
use XSLoader;

our $VERSION = '0.11';
XSLoader::load('AI::MaxEntropy', $VERSION);

sub new {
    my $class = shift;
    my $self = {
       smoother => {},
       optimizer => { },
       @_,
       samples => [],
       x_bucket => {},
       y_bucket => {},
       x_list => [],
       y_list => [],
       x_num => 0,
       y_num => 0,
       f_num => 0
    };
    return bless $self, $class;
}

sub see {
    my ($self, $x, $y, $w) = @_;
    $w = 1 if not defined($w);
    my ($x1, $y1) = ([], undef);
    # preprocess if $x is hashref
    $x = [
        map {
	    my $attr = $_;
	    ref($x->{$attr}) eq 'ARRAY' ? 
	        map { "$attr:$_" } @{$x->{$attr}} : "$_:$x->{$_}" 
        } keys %$x
    ] if ref($x) eq 'HASH';
    # convert x from strings to IDs
    for (@$x) {
        my $x_id = $self->{x_bucket}->{$_};
	# new x
	if (!defined($x_id)) {
	    push @{$self->{x_list}}, $_;
	    $self->{x_num} = scalar(@{$self->{x_list}});
	    $x_id = $self->{x_num} - 1;
	    $self->{x_bucket}->{$_} = $x_id;
	    push @$x1, $x_id;
	}
        # old x
	else { push @$x1, $x_id }
    }
    # convert y from string to ID
    my $y_id = $self->{y_bucket}->{$y};
    # new y
    if (!defined($y_id)) {
        push @{$self->{y_list}}, $y;
	$self->{y_num} = scalar(@{$self->{y_list}});
	$y_id = $self->{y_num} - 1;
	$self->{y_bucket}->{$y} = $y_id;
	$y1 = $y_id;
    }
    # old y
    else { $y1 = $y_id }
    # add the sample
    push @{$self->{samples}}, [$x1, $y1, $w];
    # update f_num
    $self->{f_num} = $self->{x_num} * $self->{y_num};
}

sub forget_all {
    my $self = shift;
    $self->{samples} = [];
    $self->{x_bucket} = {};
    $self->{y_bucket} = {};
    $self->{x_num} = 0;
    $self->{y_num} = 0;
    $self->{f_num} = 0;
    $self->{x_list} = [];
    $self->{y_list} = [];
}

sub learn {
    my $self = shift;
    $self->{lambda} = [];
    @{$self->{lambda}} = map { 0 } (1 .. $self->{f_num});
    my $o = Algorithm::LBFGS->new(%{$self->{optimizer}});
    $o->fmin(\&_neg_log_likelihood, $self->{lambda},
        $self->{progress_cb}, $self);
    my $model = AI::MaxEntropy::Model->new;
    $model->{$_} = ref($self->{$_}) eq 'ARRAY' ? [@{$self->{$_}}] :
                   ref($self->{$_}) eq 'HASH' ? {%{$self->{$_}}} :
		   $self->{$_}
        for qw/x_list y_list lambda x_num y_num f_num x_bucket y_bucket/;
    return $model;
}

1;

__END__

=head1 NAME

AI::MaxEntropy - Perl extension for learning Maximum Entropy Models

=head1 SYNOPSIS

  use AI::MaxEntropy;

  # create a maximum entropy learner
  my $me = AI::MaxEntropy->new; 
  
  # the learner see 2 red round smooth apples
  $me->see(['round', 'smooth', 'red'] => 'apple' => 2);
  
  # the learner see 3 yellow long smooth bananas
  $me->see(['long', 'smooth', 'yellow'] => 'banana' => 3);

  # and more

  # samples needn't have the same numbers of active features
  $me->see(['rough', 'big'] => 'pomelo');

  # the order of active features is not concerned, too
  $me->see(['big', 'rough'] => 'pomelo');

  # ...

  # okay, let it learn
  my $model = $me->learn;

  # then, we can make prediction on unseen data

  # ask what a red thing is most likely to be
  print $model->predict(['red'])."\n";
  # the answer is apple, because all red things the learner have ever seen
  # are apples
  
  # ask what a smooth thing is most likely to be
  print $model->predict(['smooth'])."\n";
  # the answer is banana, because the learner have seen more smooth bananas
  # (weighted 3) than smooth apples (weighted 2)

  # ask what a red, long thing is most likely to be
  print $model->predict(['red', 'long'])."\n";
  # the answer is banana, because the learner have seen more long bananas
  # (weighted 3) than red apples (weighted 2)

  # print out scores of all possible answers to the feature round and red
  for ($model->all_labels) {
      my $s = $model->score(['round', 'red'] => $_);
      print "$_: $s\n";
  }
  
  # save the model
  $model->save('model_file');

  # load the model
  $model->load('model_file');

=head1 DESCRIPTION

Maximum Entropy (ME) model is a popular machine learning approach, which
is used widely as a general classifier.
A ME learner try to recover the real probability distribution based on 
limited number of observations, by applying the principle of maximum 
entropy. The principle of maximum entropy assumes nothing on unknown data,
in another word, all unknown things are as even as possible, which makes
the entropy of the distribution maxmized.

=head2 Samples

In this module, each observation is abstracted as a sample.
A sample is denoted as C<x =E<gt> y =E<gt> w>,
which consists of a set of active features (array ref x),
a label (scalar y) and a weight (scalar w).
The client program adds a new sample to the learner by L</see>.

=head2 Features and Active Features

The features describe which characteristics things can have. And, if 
a thing has a certain feature, we say that feature is active in that
thing (an active feature). For example, an apple is red, round and smooth, 
then the active features of an apple can be denoted as an array ref
C<['red', 'round', 'smooth']>. Each element here is an active feature
(generally, denoted by a string), and the order of active features is
not concerned.

=head2 Label

The label denotes the name of the thing we describe. For the example
above, we are describing an apple, so the label can be C<'apple'>.

=head2 Weight

The weight can be simply taken as how many times a thing with certain
characteristics occurs, or how persuasive it is. For example, we see 2 
red round smooth apples, we denote it as
C<['red', 'round', 'smooth'] =E<gt> 'apple' =E<gt> 2>.

=head2 Model

After seeing enough samples, a model can be learnt from them by calling
L</learn>, which generates an L<AI::MaxEntropy::Model> object. A model is
generally considered as a classifier. When given a set of features,
one can ask the model which label is most likely to come with these
features by calling L<AI::MaxEntropy::Model/predict>.

=head1 FUNCTIONS

NOTE: This is still an alpha version, the APIs may be changed
in future versions.

=head2 new

Create a Maximum Entropy learner. Optionally, initial values of properties
can be specified here.

  my $me1 = AI::MaxEntropy->new;
  my $me2 = AI::MaxEntropy->new(
      optimizer => { epsilon => 1e-6 });
  my $me3 = AI::MaxEntropy->new(
      optimizer => { m => 7, epsilon => 1e-4 },
      smoother => { type => 'gaussian', sigma => 0.8 }
  );

The properties values specified in creation time can be changed later, like,

  $me->{optimizer} = { epsilon => 1e-3 };
  $me->{smoother} = {};

=head2 see

Let the Maximum Entropy learner see a new sample.
The weight can be omitted, in which case, default weight 1.0 will be used.

  my $me = AI::MaxEntropy->new;

  # see a sample with default weight 1.0
  $me->see(['a', 'b'] => 'p');
  
  # see a sample with specified weight 0.5
  $me->see(['c', 'd'] => 'q' => 0.5);

The sample can be also represented in the attribute-value form, which like

  $me->see({color => 'yellow', shape => 'long'} => 'banana');
  $me->see({color => ['red', 'green'], shape => 'round'} => 'apple');

Actually, the two samples above are converted internally to,

  $me->see(['color:yellow', 'shape:long'] => 'banana');
  $me->see(['color:red', 'color:green', 'shape:round'] => 'apple');

=head2 forget_all

Forget all samples the learner have seen previously.

=head2 learn 

Learn a model from all the samples,
returns an L<AI::MaxEntropy::Model> object, which can be used to make
prediction on unseen data.

  ...

  my $model = $me->learn;

  print $model->predict(['x1', 'x2', ...]);

=head1 PROPERTIES

=head2 optimizer

The optimizer is the essential component of this module. This property
enable the client program to customize the behavior of the optimizer. It
is a hash ref, containing all parameters that the client program want to
pass to the L-BFGS optimizer. Please refer to
L<Algorithm::LBFGS/List of Parameters> for details.

=head2 smoother

The smoother is a solution to the over-fitting problem. 
This property chooses the which type of smoother the client program want to
apply and sets the smoothing parameters. 

Only one smoother have been implemented in this version, 
the Gaussian smoother.

One can apply the Gaussian smoother as following,

  my $me = AI::MaxEntropy->new(
      smoother => { type => 'gaussian', sigma => 0.6 }
  );

The Gaussian smoother has one parameter sigma, which indicates the strength
of smoothing. Usually, sigma is a positive number no greater than 1.0. The
strength of smoothing grows as sigma getting close to 0.

=head2 progress_cb

Usually, learning a model is a time consuming job. And the most time 
consuming part of this process is the optimization. This callback
subroutine is for people who want to trace the progress of the optimization.
By tracing it, they can do some useful output, which makes the learning
process more user-friendly. 

This callback subroutine will be passed directly to 
L<Algorithm::LBFGS/fmin>.
You can also pass a string 'verbose' to this property, 
which simply print out the progress by a build-in callback subroutine.
Please see L<Algorithm::LBFGS/progress_cb> for details.

=head1 SEE ALSO

L<AI::MaxEntropy::Model>, L<AI::MaxEntropy::Util>

L<Algorithm::LBFGS>

L<Statistics::MaxEntropy>, L<Algorithm::CRF>, L<Algorithm::SVM>,
L<AI::DecisionTree>

=head1 AUTHOR

Laye Suen, E<lt>laye@cpan.orgE<gt>

=head1 COPYRIGHT AND LICENSE

The MIT License

Copyright (C) 2008, Laye Suen

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=head1 REFERENCE

=over

=item
A. L. Berge, V. J. Della Pietra, S. A. Della Pietra. 
A Maximum Entropy Approach to Natural Language Processing,
Computational Linguistics, 1996.

=item
S. F. Chen, R. Rosenfeld.
A Gaussian Prior for Smoothing Maximum Entropy Models,
February 1999 CMU-CS-99-108.

=back
