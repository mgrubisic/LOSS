### Loss Assessment Framework based on provided Story-loss functions

The tool allows the production of loss estimation and visualization on story-basis.


**Acronyms**

EDP:    Engineering Demand Parameter

DV:     Decision Variable

DS:     Damage State

CDF:    Cumulative distribution function

LS:     Limit State

IDR:    Inter-story drift ratio or ISDR

RIDR:   Residual inter-story drift ratio or RISDR

PFA:    Peak floor acceleration

NC:     Non-collapse

NS:     Non-structural

S:      Structural

D:      Demolition

C:      Collapse

GM:     Ground motion


**Input arguments**

* ida_ file containing IDA outputs in a .pickle format

* Period file containing 1st mode period information, in any text format

* rs file containing response spectra information in a .pickle format

* slf file containing story-loss functions produced e.g. via SLFGenerator in .xlsx format


**Step-by-step procedure**

The tool relies on three performance groups, that is
* Drift-sensitive structural elements
* Drift-sensitive non-structural elements
* Acceleration-sensitive non-structural elements

1. Read input data ← *ida outputs, RS, Periods, SLF*

    	OUTPUT: IDA outputs, RS, Periods, SLFs

2. Calculate PGA values if not provided

        OUTPUT: modified IDA outputs
        
3. Get collapse fragility functions ← *IDA outputs, IDR as EDP*

        OUTPUT: Collapse probabilities, or the parameters of the fragility functions

4. Get demolition fragility functions given non-collapse ← *IDA outputs, IML range, RIDR as EDP*
 
        OUTPUT: Demolition probabilities, or the parameters of the demolition fragility functions

5. Get the SLFs ← *SLF file name*
*Note: May be obtained from SLFGenerator (saved as a .csv or .xlsx file)*

        OUTPUT: SLFs

6. Initiate repair loss calculations

7. Get fragility functions associated with any IM ← *IDA outputs, story level, EDP of interest, IML, EDP range (IDR or PFA)*

        OUTPUT: Probabilities of exceedining EDPs, or the parameters of the calculated fragility functions

8. Estimate expected losses and perform disaggregation of losses

        OUTPUT: Loss results


NOTES:

Mandatory updates once single direction loss assessment is completed

3. Drift sensitive losses, p_edp calculation for both directions, loop for each direction, X and Y before the story loop. If non-directional, apply non_directional_conversion_factor here

4. Story_loss_ratio should include the total sum of the story (so for both directions, and for non-directional components)

5. 


-1. Finally, verify that everything works fine, that the framework is looping for all components and is not missing information


