+++
title = "Callback that stop algorithm in R"
date = 2012-07-10
lastmod = 2017-10-10T15:36:18+02:00
tags = ["R", "util"]
categories = ["programming"]
draft = false
+++

Today I was making a little programming using the mathematical software R (very useful
 for statistics, by the way), for a little test.

I'm one of the authors of a Cran package ([Rmalschains](http://cran.r-project.org/web/packages/Rmalschains/index.html)) for continuous optimization, and I was testing another packages to compare results.

Comparing a particular package I realise that the API doesn't give me enough control for
the comparisons. Briefly, to compare different algorithms all of them should stop when the same
number of solutions is achieved. Unfortunately, for the DE package, the stopping criterion is the
maximum iterations number, and for one strategy (the default strategy) this number differs,
maintaining the same maximum iterations number, in function of the function to improve. I know, not
so briefly :-).

In resume, I want to pass a function to evaluate solutions to an algorithm, and that only the first
_maxEvals_ solutions could be considered. So, it should be nice that after _maxEvals_ function evaluations
the algorithm will stop.

The aim is very simple in a theorical way, but I have only the control over a callback function used by
the algorithm, and I cannot use an 'exit' function into the function, because in that case will stop the global program,
not only the current state of the algorithm.

The solution? Using these 'complex' concepts that many people think that are useless, specially my CS students :-).
Combining a call with continuation with a closure:

```R
finalFitness = callCC (function(exitFitness) {
     fitnessCheck <- function(fn, maxevals) {
          function(x) {

               if (total == 0 || total < maxevals) {
                  total <<- total +1;
                  fitness = fn(x);

                  if (total == 1 || fitness < bestFitness) {
                     bestFitness <<- fitness;
                  }

               }

               if (total >= maxevals) {
                  exitFitness(bestFitness);
               }


               fitness;
           }

      }


      fitCheck = fitnessCheck(fun$fitness, fun$maxevals)

      log <- capture.output({
          total <- 0
          result=DEoptim(fitCheck, lower, upper, control=list(itermax=fun$maxevals/NP))
      })

      exitFitness(result$optim$bestval)
})
```

I know, it is a bit confusing. callCC implement the concept of _call-with-current-continuation_
to run a code with an _exit_ function **exitFitness** that allows me to stop the run of the algorithm.
Because the function only does a run of the  algorithm (**DEOptim**), I can stop when I want.
Also, to make it more elegant, I use a closure **fitnessCheck**  that receives a function and a
maximum number of call, and it stops when the maximum calls number is achieved
(_total_ and _bestFitness_ are global variable, so the way to modify their values is using
<<- instead of the classical <- or =).

By the way, **capture.output** is a function that disables all the output of DEoptim algorithm.
