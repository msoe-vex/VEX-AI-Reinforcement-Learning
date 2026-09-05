Copyright 2026, VEX Robotics, Inc.
## 2026 - 2027
## Game Manual
## Version 2.0

---

ii
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
### © 2026 Innovation First, Inc. / VEX Robotics, Inc. All rights reserved.
This Game Manual — including all game rules, game design concepts, scoring systems, field
specifications, field element designs, robot construction parameters, tournament structures, game
design philosophy, and all other content contained herein including all related intellectual property rights
and registrations are the exclusive proprietary property of Innovation First, Inc. used under license by
VEX Robotics, Inc. This document and all contents are subject to and protected under United States and
International Copyright and Trademark Laws.
No portion of this document may be copied, reproduced, distributed, published, displayed, transmitted,
adapted, translated, incorporated into any other work, or used as the basis for any modified or derivative
version or work, - in whole or in part, in any form or by any medium, electronic or otherwise without the prior
express written permission of Innovation First, Inc. / VEX Robotics, Inc.
This Game Manual is published solely for use in connection with authorized VEX Robotics Competitions
and events. It may not be used, in whole or in part, for or in relation to any non-authorized Robotics event,
competing competition program, alternative event format, independent game design, or related activity
without the express prior written authorization of Innovation First, Inc. / VEX Robotics, Inc. Any and all
unauthorized use of this material or any portion thereof — including adaptation for use in connection with
any competing or successor robotics events, or competition program constitutes unauthorized and illegal
misappropriation of Innovation First, Inc. / VEX Robotics, Inc.’s intellectual property and will be prosecuted to
the fullest extent permitted by law.
VEX, VEX ROBOTICS, V5, VEX V5 ROBOTICS COMPETITION, VEX IQ, VEX IQ ROBOTICS COMPETITION, VEX
ROBOTICS WORLD CHAMPIONSHIP, VEXPRO, VEXCODE, VEX 123, VEX GO, VEX AIR, and associated game
names and logos (collectively VEX Trademarks) are registered trademarks and/or trademarks owned by
Innovation First, Inc. Unauthorized use of the VEX Trademarks is strictly prohibited.
Violations of this notice and Innovation First, Inc.’s rights are subject to and/or will subject the responsible
person(s) to civil and/or criminal liabilities and prosecution.

---

Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Prefix
## Changelog ## .................................................................................## v
## Quick Reference Guide ## .......................................................## vii
## Section 1 - Introduction
## Section 2 - The Game
## VEX V5 Robotics Competition Override: A Primer ## .... ## 7
## Game Design Philosophy - A Letter from the GDC ## . ## 12
## Scoring## ..................................................................................... ## 14
## Specific Game Rules ## ........................................................... ## 19
## Safety Rules ## ...........................................................................## 27
## General Rules ## ........................................................................## 28
## General Game Rules ## ...........................................................## 30
## Section 3 - Robot Skills
## Robot Skills Challenge Rules ## ...........................................## 39
## Section 4 - The Robot
## Robot ## Rules ## ............................................................................## 44
## Section 5 - The Tournament
## Tournament Rules ## ................................................................## 58
# Table of Contents

---

iv
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Section 6 - VEX U
## Game, Robot, and Tournament Rules ## ...........................## 71
## VURC Definitions## ..................................................................## 71
## Rule Modifications: Field Setup ## ......................................## 72
## Rule Modifications: Game ## .................................................## 75
## Rule Modifications: Robot Skills Challenge## ................## 76
## Rule Modifications: Tournament ## ....................................## 78
## Rule Modifications: Robot ## .................................................## 79
## Team Composition ## ..............................................................## 85
## Appendix A - Field Overview
## Game Field Introduction ## ................................................... ## A2
## Field Overview## ...................................................................... ## A2
## Game Objects & Field Bill of Materials ## ........................ ## A3
## Field Specifications Introduction ## .................................. ## A4
## Appendix B - Glossary of Terms
## Appendix C - Rule Violations
## Appendix D - Team Classifications and Student Roles

---

v
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Version 2.0 - September 3, 2026
● Revised red-border box of <SC2> to clarify intent
● Updated <SC3> to provide clarity
● Updated <SC6> to clarify intent
● Updated <SC8> to revise the Autonomous Win Point criteria for standard and Worlds-qualifying events
● Revised the Violation Notes of <SG10> with new Match Affecting calculations
● Revised <SG11> to remove custom Violation notes and to allow Match Loads to have momentary simulta-
neous contact with Robots and Drive Team Members
● Updated <SG12> to remove height restriction and replace with restriction on Placing Scoring Objects on
the Midfield Goal
● Added a new bullet to <SG13> to clarify intent
● Updated <GG4> to provide clarity
● Removed <RSC3aiv>
● Revised <VUG4> and <VUG6> for clarity
● Removed rule <VUG7> (again)
● Updated the definition of Match Load for simplicity
● Deleted the definition of Plowing
● Revised the definition of Possession so that moving objects with a concave face of the Robot does not
count as Possession
● Other minor typo / formatting fixes
Version 1.1 - August 6, 2026
● Revised <SC6> to provide clarity
● Updated <SC7> to clarify that ending the Autonomous Period in the Midfield is not included in scoring
calculations
● Updated <SG7> and Figure SG-7 to fix a typo and provide clarity
● Revised the list of example interactions in <SG9> to provide clarity
● Added a list of example interactions to <SG10> to provide clarity
● Updated <SG12> to clarify that Robots at least partially within the Midfield during the Endgame must make
an effort to limit vertical expansion
● Revised the wording of rules <G1>, <G2> and <G4> for simplicity
● Removed rule <G5>. (Pre-existing rule <G6> is now <G5>)
● Updated <GG16> for clarity
● Updated <RSC2> to clarify that Match Load Scoring Objects can be introduced at any time, but only
through red Alliance Loaders
● Updated <RSC3b> for clarity
● Added a bullet to <R28> to clarify intent
● Revised <VUG5> to clarify intent
● Removed rule <VUT6>. (Pre-existing rule <VUT7> is now <VUT6>)
● Updated the definition of Midfield to clarify intent, and correct “outer edges” to “inner edges”
● Updated figure Q-1 to make Tournament Manager scorekeeping consistent
● Updated the definition of Student for simplicity
# Changelog
# Prefix

---

vi
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
● Revised <SG9> and <SG10> Violation Notes to clarify intent
● Moved Violation Notes from Appendix C to their corresponding rules
● Other minor typo / formatting fixes
Version 1.0 - July 2, 2026
● Revised <SC4> to provide clarity
● Updated <SG2a> and its red box to clarify that the rule only applies during the Match
● Updated <SG3> to clarify Endgame expansion limits
● Updated <SG4> to penalize only intentional remove of Scoring Objects, and to consider Scoring Objects
resting on top of the Field Perimeter as being out of the Field
● Revised the wording of <SG9> to provide better interaction examples
● Updated <SG10> to clarify intent
● Rewrote <SG12> to change the hard limit on vertical expansion during the Endgame, to a soft limit with
verbal warnings
● Added rule <SG13> to introduce Load Zone protections
● Revised <R24f> to add structurally non-homogeneous plastics to list of prohibited materials
● Added bullet points to <T6> regarding custom solutions for mounting Toggles on metal Field Perimeters
and Alliance Station dimensions / positioning as permissable Field modifications
● Removed existing rule <VUG7> and rewrote it for VEX U Load Zone protections
● Added a new definition for Load Zone
● Updated Figure T-1 to properly display the parts considered to be part of the Toggle
● Added a new drawing to Appendix A to show an example of a custom Toggle mount for metal Field
Perimeters
● Updated Violation Notes for rules <SG4>, <SG9>, <SG10>, <SG12> and <SG13>
● Other minor typo / formatting fixes
Version 0.2 - June 4, 2026
● Updated a typo in the Game Design Philosophy to clarify that the Endgame is 10 seconds long
● Revised <SC2>, <SC3>, <SC4>, <SC5>, <SC8>, <SG9>, <SG10>, <RSC3>, <VUG6> and their corresponding
figures for clarity, in regards to Pins that are Placed / Scored
● Updated <R11> to clarify intent
● Updated <R23a> to allow anodizing, painting, dyeing, or otherwise changing the color of any legal VEX
parts (other than for official events held in mainland China)
● Updated <T6> to add white tape in place of blue/red tape for Load Zones as a permissible Field modification
● Revised tiebreakers in <T22>
● Added figures to Section 6 to show the starting configuration of a VEX U Head-to-Head Match
● Updated <VUG4> to clarify that both Cups and Pins may be introduced into the Loaders in VEX U Matches
● Added a note to <R11> and <VUR11> to clarify that the 55W drivetrain motor restriction does not apply to
VEX U
● Added new drawings to Appendix A for Toggle starting orientation, and AprilTag numbering locations
● Added new definitions for Plowing, Possession and Scored to Appendix B
● Other minor typo / formatting fixes
Version 0.1 - April 27, 2026
● Initial release

---

vii
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Scoring Rules
<SC1> All scoring statuses are evaluated after the Match ends
<SC2> Placed Scoring Object criteria
<SC3> Scored Pin criteria
<SC4> A Toggle is considered set to a color when it meets all of the following criteria
<SC5> Yellow Pin Ownership
<SC6> Robot Midfield criteria
<SC7> Autonomous Bonus criteria
<SC8> Autonomous Win Point criteria
Specific Game Rules
<SG1> Starting a Match
<SG2> Horizontal expansion is limited
<SG3> Vertical expansion is limited
<SG4> Keep Scoring Objects in the Field
<SG5> Each Robot gets one Pin as a Preload
<SG6> Possession is limited to a maximum of one Pin and one Cup
<SG7> Don’t cross the Autonomous Line, and don’t interfere with your opponents’ actions
<SG8> Engage with the Midfield and Autonomous Line during the Autonomous Period at your own risk
<SG9> Alliance Goals are protected
<SG10> Scoring Objects may not be removed from neutral Goals
<SG11> Match Loads may be introduced during the Match under certain conditions
<SG12> Some rules change during the Endgame period
<SG13> Load Zones are protected during the Driver Controlled Period of the Match
Safety Rules
<S1> Be safe out there
<S2> Students must be accompanied by an Adult
<S3> Each Student Team member must have a completed participant release form on file
<S4> Stay inside the Field
<S5> Wear safety glasses
General Rules
<G1> Participants must follow the Code of Conduct
<G2> Participants must follow the Student Centered Policy
<G3> Use common sense
<G4> Students must meet the Student Eligibility Policy requirements
<G5> There is a difference between accidentally and willfully violating a Robot rule
# Quick Reference Guide

---

viii
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
General Game Rules
<GG1> Only Drive Team Members, and only in the Alliance Station
<GG2> A Team’s Robot should attend every Match
<GG3> Robots on the Field must be ready to play
<GG4> Hands out of the Field
<GG5> Match replays are allowed, but rare
<GG6> Disqualifications
<GG7> Time-outs
<GG8> Keep your Robots together
<GG9> Don’t hook your Robot to the Field, and don’t get Entangled
<GG10> The red Alliance may choose to place last
<GG11> Controllers must stay connected to the Field
<GG12> Autonomous means “no humans”
<GG13> All rules still apply in the Autonomous Period
<GG14> Don’t destroy other Robots. But, be prepared to encounter defense
<GG15> Offensive Robots get the “benefit of the doubt” when judgment calls are required
<GG16> You can’t force an opponent into a penalty
<GG17> No Holding for more than a 3-count
<GG18> Use Scoring Objects to play the game
Robot Skills Challenge Rules
<RSC1> Standard rules apply in most cases
<RSC2> Match play is different in Robot Skills Matches
<RSC3> Scoring Robot Skills Matches
<RSC4> Field setup for Robot Skills Matches
<RSC5> Skills Stop Time
Robot Rules
<R1> One Robot per Team
<R2> Robots must pass inspection
<R3> Robots must fit within an 18” x 18” x 18” volume
<R4> Officially registered Team numbers must be displayed on Robot License Plates
<R5> Let go of Scoring Objects after the Match
<R6> Robots have one Brain
<R7> Keep the power button or battery connection accessible
<R8> Firmware up to date
<R9> Use a “Competition Template” for programming
<R10> Motors are limited
<R11> Subsystems 1 & 2 have a combined motor limit
<R12> Electrical power comes from VEX batteries only
<R13> Robots use VEXnet
<R14> Give the radio some space

---

ix
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<R15> One or two Controllers per Robot
<R16> Robots are built from the VEX V5 system
<R17> New VEX parts are legal
<R18> Prohibited Items
<R19> Certain non-VEX components are allowed
<R20> Custom V5 Smart Cables are allowed
<R21> A limited amount of tape is allowed
<R22> Certain non-VEX fasteners are allowed
<R23> Visual decorations are allowed
<R24> A limited amount of custom plastic is allowed
<R25> Pneumatics are limited
<R26> The VEX Pressure Gauge is a required part if a Robot includes pneumatics
<R27> Most modifications to non-electrical components are allowed
<R28> No modifications to electronic or pneumatic components are allowed
Tournament Rules
<T1> Head Referees have ultimate and final authority on all gameplay and Robot ruling decisions
<T2> Head Referees must be qualified
<T3> Drive Team Members are permitted to immediately appeal a Head Referee’s ruling
<T4> The Event Partner has ultimate authority regarding all non-gameplay decisions
<T5> Be prepared for minor Field variance
<T6> Fields may be repaired at the Event Partner’s discretion
<T7> Fields at an event must be consistent with each other
<T8> There are three types of Field control that may be used
<T9> There are two types of Field Perimeter that may be used
<T10> Qualification Matches follow the Match Schedule
<T11> Each Team will have at least six Qualification Matches
<T12> Qualification Matches contribute to a Team’s ranking for Alliance Selection
<T13> Qualification Matches tiebreakers
<T14> Small Tournaments have fewer Alliances
<T15> Send a Student representative to Alliance Selection
<T16> Each Team may only be invited once to join one Alliance
<T17> Elimination Matches follow the Elimination Bracket
<T18> Elimination Matches are a blend of “Best of 1” and “Best of 3”
<T19> Ties in Elimination Matches lead to limited rematches
<T20> Skills Match Schedule
<T21> Skills Challenge Fields do not require the same modifications as the Head-to-Head Fields
<T22> Skills Rankings at events
<T23> Skills Rankings globally
<T24> Robot Skills at League Events

---

x
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
VEX U Game Rules
<VUG1> Different Robot starting sizes
<VUG2> Different Robot placement than rule <GG10>
<VUG3> Some electronic devices may be in motion or moving at the beginning of the Match
<VUG4> Different availability of Loaders
<VUG5> Different scoring during the Autonomous Period
<VUG6> Different Autonomous Win Point criteria
VEX U Robot Skills Challenge Rules
<VURS1> VEX U Robot Skills Matches are set up differently than V5RC Robot Skills Matches
<VURS2> Teams are permitted to use both Robots in their VEX U Robot Skills Matches
<VURS3> Both Robots must start the Robot Skills Match in legal starting positions for the red Alliance
VEX U Tournament Rules
<VUT1> VURC Head-to-Head Matches will be played 1-Team vs. 1-Team
<VUT2> Qualification Matches will be conducted in the same manner as in a V5RC Tournament
<VUT3> Elimination Matches will be conducted in the same manner, but without an Alliance Selection
<VUT4> The Autonomous Period at the beginning of each Head-to-Head Match will be 30 seconds
<VUT5> The Driver Controlled Period is shortened to 90 seconds
<VUT6> VURC Tournaments have fewer Teams in Elimination Matches
VEX U Robot Rules
<VUR1> Teams may use two (2) Robots in each Match
<VUR2> Teams may use any official VEX Robotics products, other than the exceptions listed below
<VUR3> Fabricated Parts may be made by applying the following processes to Raw Stock
<VUR4> Fabricated Parts must be made from legal Raw Stock
<VUR5> The following material types are not considered Raw Stock
<VUR6> Fabricated Parts cannot be made from Raw Stock which poses a safety or damage risk
<VUR7> Fabricated Parts must be made by Team members
<VUR8> Teams may use commercially-available springs on their Robots
<VUR9> Teams may use commercially available fastener hardware on their Robot
<VUR10> Each Robot must utilize exactly one (1) V5 Robot Brain and at least one (1) V5 Robot Radio
<VUR11> There is no restriction on the number of V5 Smart Motors
<VUR12> There is no restriction on Sensors, External Processors, or Additional Electronics
<VUR13> Commercially available Electromechanical Assemblies are not legal
<VUR14> Teams may utilize an unlimited amount of the following pneumatic components
<VUR15> Teams may use commercially available bearings on their Robot

---

Copyright 2026, VEX Robotics, Inc.
## Section 1
## Introduction

---

2
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## About the Game Manual
# Section 1 - Introduction
The VEX V5 Robotics Competition Game Manual is the most authoritative source of information for the VEX V5
Robotics Competition. Contained within this document are all of the rules, boundaries, constraints, and other
relevant information you will need to properly understand and play this year’s game.
This manual is:
● A technical reference document for gameplay, Robot construction, and event operation
● A binding set of rules for Teams, coaches, referees, Event Partners, and all other volunteers and
participants to adhere to
● The primary source of truth for all things related to the VEX V5 Robotics Competition
This manual is not:
● A strategy guide to score the most points
● An instruction set to build the best Robot
● A replacement for referee training, Event Partner training, or other event procedure
## How to Read the Game Manual
The rules in this game manual are intended to work together to create a system of constraints. These rules are
not intended to be understood independently of one another or in isolation from other information contained
in this document. Many situations may require Teams, referees, or other volunteers to use logic from multiple
places in the manual to properly form an interpretation. It is critical that the entire manual is read and under-
stood, not just parts.
Information in the game manual is presented in the following ways:
Definitions in Appendix B establish the meaning of a term with regards to this document. Sometimes, these
definitions may not perfectly match a more commonly accepted dictionary definition. In that case, the VEX
definition takes precedence. If there is no VEX definition for a term, it can be reasonably assumed that a dictio-
nary definition can be used.
General Rules establish the baseline rules for competition that Teams must adhere to at all times. These
include, but are not limited to, conduct at an event, roster eligibility, competition integrity, and authority and
enforcement.
General Game Rules begin to outline the rules that Teams must follow in every VEX game, not just the rules
specific to this season’s game. Many of these rules do not change from year to year, and help prevent the game
from devolving into immediate chaos.
Scoring Rules define how points are earned and evaluated.
Specific Game Rules describe what Robots and Drive Team Members may and may not do during a Match,
specifically for this season’s game. These rules are subject to change from year to year, depending on how the
game is designed and meant to be played.

---

3
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Game Manual Updates
Robot Rules define how Robots may be built and configured.
Tournament Rules describe how competitions are run, and how Teams are ranked at events.
Some rules also reference Violation Notes that are located in Appendix C. These notes provide additional
guidance on enforcement, escalation, or special circumstances. If a rule doesn’t include a cross-reference to
Violation Notes, standard Violation definitions apply. See Appendix C for information about rule Violations and
penalties.
There are also red-bordered boxes of text placed in some areas of the game manual. These are
intended to provide further clarification and guidance in places that the Game Design Commit-
tee has deemed may benefit from things being said a different way, or presented slightly differ-
ently. Red-border boxes are meant to be supplements to, not replace, rules or definitions.
This manual will have a series of “major” and “minor” updates over the course of the season. Each version is
official and must be used in official V5RC events until the release of the next version, upon which the previous
version becomes void.
The latest version of the Game Manual can always be found at https://link.vex.com/docs/26-27/v5rc/
game-manual.
Known major release dates are as follows:
Release Date Effective Date Version # Details
April 27, 2026 April 27, 2026 Version 0.1 Initial game release
May 14, 2026 May 14, 2026 N/A Official Q&A system opens
June 4, 2026 June 11, 2026 Version 0.2
Minor typographical errors or formatting
issues found in the initial release, with
very few rule changes expected
July 2, 2026 July 9, 2026 Version 1.0
May include gameplay or rule changes
inspired by input from the official Q&A
system and the VEX community
August 6, 2026 August 13, 2026 Version 1.1 Clarification / minor update
September 3, 2026 September 10, 2026 Version 2.0 
May include gameplay or rule changes
inspired by early-season events
October 8, 2026 October 15, 2026 Version 2.1 Clarification / minor update
December 3, 2026 December 10, 2026 Version 2.2 Clarification / minor update
January 28, 2027 February 4, 2027 Version 3.0 
May include gameplay or rule changes
inspired by mid-season events
March 25, 2027 April 1, 2027 Version 4.0
May include gameplay or rule changes
pertaining specifically to the VEX
Robotics World Championship

---

4
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## The Q&A System
In addition to these known major updates, there may also be unscheduled updates released throughout the
season if deemed critical by the Game Design Committee.
Any scheduled or unscheduled updates will always be released on a Thursday, no later than 5:00 PM CST
(11:00 PM GMT). These updates will be announced via the VEX Forum, automatically pushed to the VEX V5 Hub
app, and shared via VEX Robotics social media & email marketing channels. Once announced, the new version
of the Game Manual will be immediately available at the link above.
Generally, Game Manual updates, scheduled or unscheduled, will include a grace period before the updated
rules go into effect for competitions. See the release table above for specific dates. This grace period does not
apply to the Version 0.1 Release, which serves as the initial rule set for the season.
Any events that begin before the 7-day grace period has ended must continue using the rules from the
previous Game Manual Release. This policy ensures fairness and consistency, allowing Teams to adapt their
strategies and gameplay accordingly before the changes are enforced in official competitions.
Once a manual update occurs, the previous version will be re-uploaded with an “Obsolete” watermark. Those
will be found at https://link.vex.com/docs/26-27/v5rc/game-manual-obsolete and will be available to reference
through the rest of the season.
The Game Design Committee reserves the right to enforce critical updates to the Game Manual as effective
immediately upon release, if we feel that the changes are critical for competitive integrity, safety, and/or other
extenuating circumstances.
Multi-week league events (or similar) that cross over a grace period should use the version of the Game Manual
that is in effect at the beginning of each league session. Leagues should update to new versions of the Game
Manual between sessions as appropriate.
When first reviewing a new robotics game, it is natural to have questions about situations which may not be
immediately clear. Navigating the Game Manual and seeking out answers to these questions is an important
part of learning a new game. In many cases, the answer may just be in a different place than you first thought -
or, if there is no rule explicitly prohibiting a gameplay strategy, then that usually means it is legal!
However, if a Team is still unable to find an answer to their question after closely reviewing the relevant rules,
then every Team has the opportunity to ask for official rules interpretations and clarifications in the VEX
Robotics Question & Answer System. These questions may be posted by a Team’s Adult representative via the
events.vex.com account that is associated with that Team.
All responses in this Q&A system should be treated as official rulings from the VEX Robotics Game Design
Committee, and they represent the correct and official interpretation of the VEX V5 Robotics Competition
rules. The Q&A system is the only source besides the Game Manual for official rulings and clarifications, and is
functionally an extension of the Game Manual. Unlike Game Manual updates, Q&A rulings are effective immedi-
ately upon release.
The VEX V5 Robotics Competition Question & Answer System will open May 14th, 2026.

---

VEX V5 Robotics Competition Override - Game Manual
5
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Before posting on the Q&A system, be sure to review the Q&A Usage Guidelines:
1. The Q&A system is for rules clarifications only.
2. Only registered Teams, certified Event Partners, and certified V5RC Head Referees can post questions.
3. Read and search this Game Manual before posting.
4. Read and search existing Q&As before posting.
5. Quote the applicable rule from the latest version of this Game Manual in your question.
6. Make a separate post for each question.
7. Use specific and appropriate question titles.
8. Questions will (mostly) be answered in the order they were received.
9. This system is the only source for official rules clarifications.
10. The Game Design Committee cannot and will not overrule a Head Referee’s decision.
If there are any conflicts between the English-language PDF of the Game Manual and other supplemental
or translated materials (e.g., referee training materials, the V5RC Hub app, the game reveal video, a trans-
lated game manual, etc.), the most current version of the English-language PDF of the Game Manual takes
precedence.
Similarly, it can never be assumed that definitions, rules, or other materials from previous seasons apply to the
current game. Q&A responses from previous seasons are not considered official rulings for the current game.
Any relevant clarifications that are needed should always be re-asked in the current season’s Q&A.
## Hierarchy of Information
This Game Manual is the authoritative source for all gameplay, Robot, and event rules. For these matters,
official information ranks as follows:
1. The current version of this Game Manual, including all published updates
2. The official Q&A system
3. Other VEX Robotics or Global Robotics and Science Foundation supplementary documents and training
materials
Q&A answers are official rulings, effective when posted. An answer that conflicts with this Manual takes
precedence until a Manual update addresses it. Answers not addressed by an update remain in effect.
The following companion documents are authoritative within their own scopes and take precedence over any
summary or restatement of their content in this Manual:
● Code of Conduct
● Student-Centered Policy
No other source of information is official.

---

Copyright 2026, VEX Robotics, Inc.
## Section 2
## The Game

---

7
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Section 2 - The Game
# VEX V5 Robotics Competition Override: A Primer
VEX V5 Robotics Competition Override is played on a 12’ x 12’ square Field, set up as illustrated in the figures
throughout this manual.
In Head-to-Head Matches, two (2) Alliances — one (1) “red” and one (1) “blue” — composed of two (2) Teams
each, compete in Matches consisting of a fifteen (0:15) second Autonomous Period followed by a one minute
and forty-five second (1:45) Driver Controlled Period.
The object of the game is to attain a higher score than the opposing Alliance by stacking Pins and Cups on
Goals, setting Toggles to your Alliance color, and ending the Match with your Robot in the contested Midfield.
An Autonomous Win Point is awarded to any Alliance that completes a set of assigned tasks by the end of the
Autonomous Period.
An Autonomous Bonus is awarded to the Alliance that has the most points at the end of the Autonomous
Period.
Teams may also compete in Robot Skills Matches, where one (1) Robot tries to score as many points as
possible. See Section 3 for more information.
At the VEX U Collegiate level, Teams play in a modified Tournament with a 30-second Autonomous Period and
additional Robot build challenges. See Section 6.

---

8
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Field Overview
The V5RC Override Field consists of the following:
● 56 Cups
० 20 Match Loads
■ 10 for red Alliance
■ 10 for blue Alliance
० 24 that start the Match in predetermined locations on the Field (gray side up)
० 12 that start the Match in predetermined locations on the Field (clear side up)
● 63 Pins
० 4 red/blue that start the Match in predetermined locations
० 20 red/yellow
■ 2 Preloads
■ 10 Match Loads
■ 8 that start the Match in predetermined locations
० 20 blue/yellow
■ 2 Preloads
■ 10 Match Loads
■ 8 that start the Match in predetermined locations
० 19 yellow/yellow
■ 2 Match Loads
■ 17 that start the Match in predetermined locations
● 9 Goals
० 4 Alliance colored
■ 2 red
■ 2 blue
० 5 neutral colored
■ 4 short
■ 1 tall
● 4 Toggles
● 4 Loaders, two adjacent to each Alliance Station
Note: The illustrations in this section of the Game Manual are intended to provide a general visual un-
derstanding of the game. Some figures may highlight or change the appearance of certain Field
Elements and Scoring Objects to emphasize or clarify intent.
Teams should refer to official Field specifications, found in Appendix A, for exact Field dimensions, a full
Field bill of materials, and exact details of Field construction.

---

9
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure FO-1: An overhead view of the V5RC Override Field, with Alliance Stations (orange),
Loaders (white), Toggles (pink), and Goals (green) highlighted.

---

10
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure FO-2: An overhead view of the V5RC Override Field in its starting configuration, with icons representing orientation of Scoring Objects

---

11
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure FO-3: The recommended locations of the Head Referee (black & white stripes), and Scorekeeper Referees (black & white checkerboard).

---

12
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Game Design Philosophy - A Letter from the GDC
We want to take the opportunity to explain how this game was designed and what kind of gameplay we expect
to see during the season. This section also points out parts of the game that we will watch closely in case
updates are needed.
This is not a rules section. It does not add any new rules. Instead, it helps you understand the purpose of the
game and what we as designers had in mind.
What This Game is About & Scoring
Override is a game that focuses on stacking and teamwork. We intend for Teams to work together with their
Alliance partners to assemble stacks, gain points using Scoring Objects, and control Toggles to outscore their
opponents.
Because of this, success in the game is not just about building a good Robot. It also depends on how well you:
● Communicate with your partner
● Plan your strategy
● Time your actions during a Match
Teams that coordinate well and make smart decisions will have an advantage.
The game is designed so that most points come from constructive actions, such as:
● Building stacks
● Flipping Toggles
● Positioning your Robot effectively
Interaction with the opposing Alliance is still an important part of the game. However, Matches cannot be
dominated by tearing down what the other Alliance has built. Some defense and disruption is expected, but the
main focus should stay on building and scoring.
If strategies that focus mostly on preventing stacks become too common, the Game Design Committee may
consider making changes to ensure the game remains interesting with many strategy choices. We will look at
how often these strategies are used and how they affect gameplay compared to our intentions.
Large Point Swings with Toggles and Yellow Pins
The scoring values in Override are designed to encourage dynamic strategy and decision making. Because of
this, it is possible for actions like controlling Toggles or scoring yellow Pins to create large point swings during
a Match.
These moments can be exciting, but they should not become the only way to win. Teams should still have
multiple viable strategies, including building stacks and competing in the Endgame.

---

13
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
We will monitor how these point swings impact Matches throughout the season. If they begin to outweigh other
scoring opportunities or reduce strategic variety, the Game Design Committee may adjust point values to
maintain a balanced and engaging game.
The Endgame: Final 10 Seconds
The last 10 seconds of the Match are designed to feel more intense and competitive. During this time, the
game shifts to a “king of the hill” style challenge in the Midfield. This section of the Match will likely include
increased Robot interaction. As such:
● Positioning and timing become critical
● Robot durability matters more
Teams should expect more crowded and contested conditions than earlier in the Match.
If the Endgame consistently leads to problems like Robots getting stuck, tipping too often, or not being able to
recover, the Game Design Committee may adjust rules to keep Matches fair and playable.
Using Sensors in the Game
Override is a game where sensors can be very helpful. Tools like AI Vision Sensors and Optical Sensors can
help your Robot:
● Identify objects
● Interact with Toggles
● Navigate the Field
The Game Design Committee will work to support sensor use through Field and game design. However, real-
world conditions are not always perfect. Lighting and object placement may vary. Because of this, Teams
should design and code their Robots so their sensors can handle small changes and still work reliably during a
Match.
In addition, Robot rules will continue to be reviewed to make sure the game stays fair throughout the season.
One important focus is preventing Teams from building Robots that could interfere with sensors in unfair ways
like intentionally trying to confuse sensors or adding parts that look like Field Elements.
Making sure all Teams can rely on their sensors to work as expected is an important part of fair competition.
Have a great season!
- The VEX V5 Robotics Competition Game Design Committee

---

14
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Scoring
Autonomous Bonus 12 Points
Each Scored Alliance-colored Pin 5 Points
Each Scored yellow Pin 10 Points
Each Robot ending in the Midfield 8 Points
<SC1> All scoring statuses are evaluated after the Match ends. Scores are calculated five seconds after
the Match ends, or once all Scoring Objects, Field Elements, and Robots on the Field come to rest, whichever
comes first.
a. This 5-second delay is intended to be the only permitted “benefit of the doubt” for last-second scoring
actions. If an object or Robot is still in motion and “too close to call” between two states at the 5-second
mark, then the less advantageous of the two states should be awarded to the Robot(s) in question. A Robot
which is breaking the plane of the Midfield but slowly droops down and away from the Midfield at five (5)
seconds would not be considered in the Midfield.
b. At the end of the Match, the on-screen timer displayed by Tournament Manager will hold the current Match
information and “0:00” for five (5) seconds before moving to queue the next Match. This should be the
primary 5-second visual cue used by Teams and Head Referees.
c. This 5-second delay is only intended to be a “benefit of the doubt” grace period, not an extra five seconds
of Match time. Robots which are designed to strategically exploit this grace period will receive a Minor
Violation, and any post-Match movement will not be included in score calculation (i.e., the Match will be
scored as it was at 0:00).
d. Referees should avoid contacting or moving Robots and/or Scoring Objects as much as possible while
evaluating scoring statuses. If an object must be moved to evaluate the status of another object, its status
must be agreed upon by all Teams and the Head Referee, and noted or recorded, before it is moved.
e. Referees must record counts based on verified scoring statuses evaluated after the Match, using final
positions of Scoring Objects, Field Elements, and Robots. Point considerations used to determine whether
a Violation is Match Affecting (e.g., specified in Violation Notes) should NOT be added to or deducted from
the actual score, and points scored during a Violation should not be deducted from a score.
<SC2> Placed Scoring Object criteria.
a. A Pin is considered Placed if it meets all of the following criteria:
i. The Pin is partially or entirely nested with a Goal, or with a Cup that is partially or entirely nested with
another Placed Pin.
ii. Each Goal and/or each half of a Cup nested with that Pin contains a maximum of one Pin half. (See
Figure SC2-2.)
b. A Cup is considered Placed if it meets the following criteria:
i. The Cup is partially or entirely nested with a Placed Pin.

---

15
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<SC3> Each Pin consists of two halves. Each visible half of a Placed Pin counts as a Scored Pin, according
to the following:
a. Each visible red half of a Placed Pin counts as a Scored red Pin, earning points for the red Alliance.
b. Each visible blue half of a Placed Pin counts as a Scored blue Pin, earning points for the blue Alliance.
c. Each visible yellow half of a Placed and Owned Pin counts as a Scored yellow Pin, earning points for the
Alliance that Owns the Pin. (See <SC5>.)
For a Pin half to be considered visible, the Pin half must not be partially or fully nested inside the opaque half of
a Cup.
<SC4> A Toggle is considered set to a color when it meets all of the following criteria:
a. The Toggle must be fully seated, such that there is a face of the Toggle in contact and parallel with its
mounts on the Field Perimeter at rest. (see Figure SC5-1)
b. The Toggle is not in contact with a Robot from either Alliance.
If a Toggle is not considered set to a color, it is considered a neutral (yellow) Toggle by default, and neither
Alliance receives Ownership of the yellow Pins Placed in that Quadrant. While the Toggle has infinite potential
orientations, only three discrete orientations are considered “set” states.
Figure SC2-1: The Scoring Objects stacked in the Goal
all count as Placed, as they are all at least partially nested
with the Goal and/or other Placed Scoring Objects.
Figure SC2-2: The Pin resting on top is partially nested
with the Cup, but the Pin is not considered Placed since
that half of the Cup contains more than one Pin half.
In the context of <SC2>, nested means that one half of a Pin is partially or completely contained
within the inner volume of a Cup or Goal. In other words, the Pin is breaking an imaginary plane
at the opening of a Cup or Goal, in any way. Robot contact does not matter for the purposes
of <SC2>, and a Robot that is contacting or Possessing a Pin and/or Cup that still meets the
criteria presented in <SC2> does not negate the Placed status, provided no other rules are
broken (particularly <SG6>).

---

16
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<SC5> Yellow Pin Ownership. Each Placed Pin with one or more yellow halves can be Owned by an Alliance.
a. A yellow Pin Placed in a Quadrant is Owned by an Alliance if the Toggle in that Quadrant is set to the
Alliance’s color. If the Toggle is set to yellow, Placed yellow Pins in that Quadrant are not Owned and do not
score points. (See <SC3>.)
b. A yellow Pin Placed in the Midfield is Owned by the Alliance that ends the Match with a greater number of
Robots in the Midfield. (See <SC6>.)
i. If both Alliances end the Match with an equal number of Robots in the Midfield, yellow Pins Placed in the
Midfield are not Owned and do not score points.
Figure SC5-1: This Quadrant’s Toggle is set to red, so yellow Pins
Placed in this Quadrant’s Goals are Owned by the red Alliance.
Figure SC5-2: The blue Alliance has more Robots in the
Midfield at the end of the Match, so yellow Pins Placed in
the Midfield Goal are Owned by the blue Alliance.

---

17
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Toggle: YellowYellow Toggle: Blue Toggle: Red
Red Pins: 15 points Red Pins: 15 Points Red Pins: 15 Points
Blue Pins: 5 Points Blue Pins: 5 Points Blue Pins: 5 Points
YellowYellow Pins: 0 Points 
YellowYellow Pins: 30 Points
(Scored for Blue)
YellowYellow Pins: 30 Points
(Scored for Red)
Total
Red: 15 Points / Blue: 5 Points
Total
Red: 15 Points / Blue: 35 Points
Total
Red: 45 Points / Blue: 5 Points
<SC6> A Robot counts as ending in the Midfield if the Robot is at least partially within the Midfield at the end
of the Match (i.e., any part of the Robot extends into the defined volume of the Midfield).
<SC7> Scoring of the Autonomous Bonus is evaluated immediately after the Autonomous Period ends.
a. Scoring methods that rely on Robots’ end-of-Match positioning in the Midfield (e.g., Robots ending in the
Midfield and Ownership of yellow Pins in the Midfield) are not included in Autonomous Period scoring
calculations.
b. If the Autonomous Period ends in a tie, including a zero-to-zero tie, each Alliance will receive an
Autonomous Bonus of six (6) points.

---

18
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
c. Any Violations, Major or Minor, committed during the Autonomous Period will result in the Autonomous
Bonus being automatically awarded to the opposing Alliance. See <GG13>.
d. Per rule <GG13>, if both Alliances commit Violations during the Autonomous Period, then no Autonomous
Bonus will be awarded.
This rule is applied differently for VEX U. See Rule <VUG5>.
<SC8> An Autonomous Win Point is awarded to any Alliance that ends the Autonomous Period with all of the
following tasks completed, and that has committed no Violations during the Autonomous Period:
1. At least six (6) Pins Scored for your Alliance. (Does not include Pins Scored in Quadrants on the opposing
side of the Autonomous Line.)
2. At least two (2) Goals each contain at least two (2) Pins Scored for your Alliance. (Does not include Goals in
Quadrants on the opposing side of the Autonomous Line.)
3. Neither Robot is contacting the Field Perimeter.
For events which qualify directly to the VEX Robotics World Championship (such as Event Region
Championships and Signature Events), the following tasks must be completed for an Alliance to receive an
Autonomous Win Point. The standard criteria above still apply to all other events.
1. At least seven (7) Pins Scored for your Alliance. (Does not include Pins Scored in Quadrants on the
opposing side of the Autonomous Line.)
2. At least three (3) Goals each contain at least two (2) Pins Scored for your Alliance. (Does not include Goals in
Quadrants on the opposing side of the Autonomous Line.)
3. Neither Robot is contacting the Field Perimeter.
Autonomous Win Point criteria may be further modified for the World Championship if needed,
with details to be released in a future Game Manual update. Autonomous Win Point criteria for
World Championship-qualifying events will be used as a baseline to determine criteria for the
World Championship. Any potential modifications would be minor.
This rule is applied differently for VEX U. See Rule <VUG6>.

---

19
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Specific Game Rules
<SG1> Starting a Match. Prior to the start of each Match, each Robot must be placed such that it meets all of
the following criteria:
a. No larger than 18” (457.2 mm) long by 18” (457.2 mm) wide by 18” (457.2 mm) tall.
b. Not contacting any Scoring Objects other than a maximum of one (1) Preload. See rule <SG5>.
c. Not contacting Goals, Loaders, Load Zones, or Toggles.
d. Not contacting any other Robots, and not sharing a Quadrant with another Robot.
e. Completely stationary (i.e., no motors or other mechanisms in motion).
f. Contacting the Field tiles and Field Perimeter on their Alliance’s side of the Autonomous Line.
Note: Using external influences, such as Preloads or the Field Perimeter, to maintain a Robot’s starting
size is only acceptable if the Robot would still satisfy the constraints of <R3> and pass inspection
without these influences.
Violation Notes:
1. The Match will not begin with any conditions in this rule unmet. If a Robot cannot meet
these conditions in a timely manner, the Robot will be removed from the Field and rules
<R2d> and <GG2> will apply until the situation is corrected. They will not receive a Disquali-
fication, but should receive a Minor Violation and will not be permitted to play in the Match.
Clause A of this rule is applied differently for VEX U. See Rule <VUG1> and <VUG3>.
Figure SG-1: An overhead view of the Field, with four Robots in legal starting positions.

---

20
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<SG2> Horizontal expansion is limited. Once the Match begins, Robots may expand horizontally beyond the
starting size limit within the following criteria:
a. The Robot can never be larger than 24” wide or 24” long at any point during the Match (must fit within a 24”
x 24” square horizontal footprint).
Teams should be aware that Robots may incidentally expand horizontally while extending verti-
cally (e.g., mechanisms that arc, swing, or deploy upward). Teams must be prepared to demon-
strate that their Robot does not exceed the maximum size constraint of 24” x 24” at any point
during a Match due to either physical or programmed limitations, including while any vertical
expansion mechanisms are in use.
Violation Notes:
Incidental/insignificant infractions that occur during a Match are only considered Minor Viola-
tions. Repeated Minor Violations should only escalate to a Major Violation in extreme circum-
stances. Examples of Minor Violations include, but are not limited to:
● Loose wires
● Broken zip ties / rubber bands
● Bent or broken mechanical components that are not used for strategic gain
<SG3> Vertical expansion is limited. Once the Match begins, Robots may expand vertically beyond the
starting size limit, but no part of the Robot may exceed an overall height of 50” at any point during the Match
(must always be able to fit within a hypothetical 50” vertical sizing box). For exceptions to this rule involving
Robots in the Midfield during the Endgame period, see <SG12>.
<SG4> Keep Scoring Objects in the Field. Teams may not intentionally remove Scoring Objects from the Field.
A Scoring Object that leaves the Field during Match play will be returned to the Field in a location near where
it left, in contact with the Field tiles and the Field Perimeter but no other Field Elements or Scoring Objects
and no Robots. Volunteers should return Scoring Objects to the Field as quickly as possible, and any delay in
returning a Scoring Object should not be considered Match Affecting or cause for a replay.
a. A Scoring Object that comes to rest on top of the Field Perimeter is considered to have left the Field.
Figure SG2: A demonstration of how the size of the Robot may change horizontally through the course of a vertical expansion.

---

21
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Violation Notes:
1. Due to the difficulty of determining Match Affecting implications of this rule, most Viola-
tions should be considered Minor. However, blatantly intentional and/or Match Affecting
Violations (especially during Elimination Matches) may still immediately escalate to a Major
Violation at the Head Referee’s discretion.
<SG5> Each Robot gets one Pin as a Preload. Red Alliance Preloads are red/yellow Pins; blue Alliance
Preloads are blue/yellow Pins. Prior to the start of each Match, each Preload must be placed such that it meets
all of the following criteria:
a. Contacting one Robot of the same Alliance color as the Preload.
b. Not contacting the same Preload as another Robot.
c. Not contacting other Scoring Objects.
d. Not contacting any other Goals, Loaders, Load Zones, or Toggles.
Note: If a Robot is not present for their Match, then that Robot’s Preload may be used as a Match Load in
accordance with <SG11>.
Violation Notes:
1. See <SG1>.
<SG6> Possession is limited to a maximum of one Pin and one Cup. Robots may not have Possession of
more than one (1) Pin at once. Robots may not have Possession of more than (1) Cup at once. Robots in Viola-
tion of this rule must immediately stop all actions except for attempting to remove the excess Scoring Objects.
If they are unable to remove the excess Scoring Objects, then they must return to a legal starting position (as
described by <SG1>). They will not be eligible to receive points for ending the Match in the Midfield, and cannot
interact with Toggles, Goals, or other Scoring Objects while in Possession of excess Scoring Objects.
<SG7> Don’t cross the Autonomous Line, and don’t interfere with your opponents’ actions. During the
Autonomous Period, Robots may not contact foam tiles, Scoring Objects, or Field Elements which are on the
opposing Alliance’s side of the Autonomous Line.
a. The Autonomous Period should be primarily Offensive, with Teams focusing on scoring and executing
strategic maneuvers rather than Defensive disruption. Teams should avoid actions that are primarily Defen-
sive in nature, including but not limited to:
i. Intentionally disrupting Scoring Objects or Field Elements on the opponent’s side of the Autonomous
Line.
ii. Deliberately contacting an opponent’s Robot to interfere with their autonomous path.
b. Scoring Objects that begin the Match on the Autonomous Line are not considered to be on either side, and
may be utilized by either Alliance during the Autonomous Period.

---

22
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure SG-7: These Scoring Objects (circled in green) would be considered to be on the Autonomous Line.
Note: All 28 Scoring Objects that begin the Match above or in contact with the Autonomous Line are con-
sidered to be on the Autonomous Line. See Figure SG-7.
c. While incidental contact or unintentional interactions may occur with Robots on the other side of the
Autonomous Line, Teams that employ deliberate Defensive autonomous strategies that impact their
opponents’ autonomous routines may be subject to Minor or Major Violations at the discretion of the Head
Referee.
d. Teams cannot intentionally move Scoring Objects to the opposing Alliance’s side of the Autonomous Line.
e. Contact with either of the following during the Autonomous Period will result in the Autonomous Bonus
and an Autonomous Win Point being awarded to the opposing Alliance, unless the opposing Alliance also
breaks rules in the Autonomous Period:
i. An opposing Robot that isn’t interacting with (1)the Autonomous Line, (2) objects that begin the Match
on the Autonomous Line, or (3) the Midfield (see <SG8>).
ii. Scoring Objects on the other side of the Autonomous Line.
Violation Notes:
1. All Violations of this rule (Major or Minor) will result in the Autonomous Bonus being
awarded to the opposing Alliance. See <SG8b> for a potential exception caused by Autono-
mous Line interactions.
2. Intentional, strategic, or egregious Violations, such as intentional contact with an opposing
Robot while contacting the foam tiles on the opposing side of the Autonomous Line, or the
interactions described in <SG7e> will be considered Major Violations and should result in a
Disqualification for the Match.
3. Deliberate Defensive Autonomous strategies, as described in <SG7a>, may also be
recorded as <G1> Violations at the Head Referee’s discretion.

---

23
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<SG8> Engage with the Midfield and/or Autonomous Line during the Autonomous Period at your own risk.
Any Robot that engages with the Midfield and/or Scoring Objects that begin the Match on the Autonomous
Line should be aware that opponent Robots may also choose to do the same. Per <GG12> and <GG13>, Teams
are responsible for the actions of their Robots at all times.
a. For the purposes of this rule, “engages with” means any combination of:
i. Contacting foam tiles within the Midfield
ii. Contacting the Goal in the Midfield
iii. Contacting Scoring Objects that begin the Match on the Autonomous Line
b. If opposing Robots contact one another while both engaging with the Midfield or the Autonomous Line, and
a possible <GG14> Violation occurs (e.g., damage, Entanglement, or tipping over), a judgment call will be
made by the Head Referee within the context of <GG14> and <GG15> (just as it would if the interaction had
occurred during the Driver Controlled Period).
c. Intentional, strategic, repeated, or egregious offenses, such as negatively impacting Robots that are not
engaging with the Midfield or the Autonomous Line, may still be deemed a Violation of <GG13>, <GG14>,
<GG15>, <SG7>, <G1>, and/or <S1> at the Head Referee’s discretion.
The Midfield and the Scoring Objects that begin on the Autonomous Line are intended to
be utilized by both Alliances during the Autonomous Period. This will inevitably result in Ro-
bot-on-Robot interactions, both incidental and intentional. The overarching intent of <SG8> is
for the vast majority of these interactions to result in no rule Violations and / or penalties for
either Alliance, just as no rules Violations occur in 99% of Driver Controlled interactions.
Teams are responsible for the actions of their Robots at all times. A Robot with a small wheel
base, which tips over every time they enter the Midfield and contacts an opponent, should not
attempt to claim a <GG14> Violation on their opponent.
With that being said, the Midfield is a neutral zone, not a “free-for-all” zone. The intent of
clause D is to provide Head Referees with the leeway to still make a judgment call, if needed,
when a Team has chosen to exploit this rule beyond its intent. Reckless or unsafe strategies
aimed solely at the destruction, damage, tipping over, Entanglement, Trapping, or forcing of an
opponent into a penalty are still prohibited in the VEX Robotics Competition.
<SG9> Alliance Goals are protected. Robots may not directly or indirectly interact with opposing Alliance-col-
ored Goals. Examples of interactions include, but are not limited to:
● Making contact with the Goal or with Scoring Objects stacked on the Goal. (Direct)
● Adding or removing Placed Scoring Objects. (Direct)
● Strategically covering the Goal in a manner that prevents additional Scoring Objects from being Placed.
(Indirect)
● Pushing an opposing Robot into Scoring Objects stacked in their own Alliance-colored Goal, knocking
some or all of them out of the Goal. (Indirect)

---

24
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Violation Notes:
1. Incidental interactions, such as unintentional / inconsequential contact with the Goal and/or
Scoring Objects stacked on the Goal, should typically be considered Minor Violations.
2. Intentional, strategic, or egregious interactions, including any interactions involving
the addition or removal of Placed Scoring Objects on the Goal, will be considered Major
Violations.
3. Repeated Minor Violations can escalate to a Major Violation at the Head Referee’s discre-
tion, especially in cases where Violations occur after multiple prior warnings.
<SG10> Placed Scoring Objects may not be removed from neutral Goals via direct or indirect interactions.
Examples of interactions include, but are not limited to:
● Grabbing and lifting a Scoring Object out of the Goal. (Direct)
● Contacting Scoring Objects stacked in the Goal, knocking some or all of them out of the Goal. (Direct)
● Pushing an opposing Robot into Scoring Objects stacked in the Goal, knocking some or all of them out of
the Goal. (Indirect)
Violation Notes:
1. For the purposes of Match Affecting calculations, the removal of Placed Scoring Objects
from neutral Goals should be evaluated as follows:
a. For an interaction resulting in the removal of 1, 2, or 3 Pins, each Pin removed as part of
this interaction should be considered worth a value of 10 points.
b. For an interaction resulting in the removal of 4+ Pins, the Pins removed as part of this
interaction should be considered worth a combined value of 50 points.
These values are not added to the actual score. If subtracting these point values from the
offending Alliance’s final score would change the outcome of the Match, then the Viola-
tion(s) should be considered Match Affecting.
Note: Head Referees cannot be expected to track the number of Pins Placed in each Goal
throughout a Match, and when Pins get removed from Goals, they cannot be expected to
have a 100% accurate count of the Pins that were previously Placed in that Goal. For each
interaction involving removed Pins, Head Referees can only be asked to make their best
educated guess as to how many Pins were removed, and Teams must accept Head Referee
counts, even if not perfectly accurate. (Accurately counting 1, 2, or 3 Pins is difficult.
Counting beyond 3 becomes even less reliable, which is why a fixed point value applies to
interactions resulting in the removal of 4+ Pins.)
2. Intentional, strategic, or egregious Violations will be considered Major Violations.
3. Repeated Minor Violations can escalate to a Major Violation at the Head Referee’s discre-
tion, especially in cases where Violations occur after multiple prior warnings.
<SG11> Drive Team Members may introduce Match Loads to the Field through Loaders, under the follow-
ing conditions:
a. Drive Team Members may only introduce Match Loads through their Alliance-colored Loaders, and only

---

25
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
during the Driver Controlled Period of the Match.
b. Match Loads may only be introduced by placing or dropping them through the top or back of a Loader,
either separately or nested together.
c. Match Loads may have momentary, simultaneous contact with both a Drive Team Member and a Robot as
the Match Load is introduced, prior to being released by the Drive Team Member. Once a Match Load has
been released, it may no longer be contacted by a Drive Team Member.
d. Match Loads may only be removed through the bottom opening of the Loader.
Excessive, unnecessary, or unsafe actions while introducing a Match Load may be considered a Violation of
<S1> and/or <G1> at the Head Referee’s discretion. Drive Team Members must be careful not to lift the Field
Perimeter while lifting the Loader.
Figure SG11-1: Scoring Objects can be introduced
through the top opening of the Loader.
Figure SG11-2: Scoring Objects may also be introduced
through the back of the Loader while it is raised.
This rule is applied differently for VEX U. See Rule <VUG4>.
<SG12> Some rules change during the Endgame period.
1. Robots may not Place Scoring Objects on the Midfield Goal during the Endgame period.
2. Robots that attempt to end the Match in the Midfield should expect vigorous interactions from opposing
Robots, and must accept increased risk of becoming Entangled or being tipped, especially when vertically
expanded. When a Robot is contacting or engaging with the Midfield, or is in proximity to the Midfield,
incidental damage that is caused by opposing Robots pushing, tipping, or becoming Entangled with them
would not be considered a Violation of <GG14>. Intentional damage or dangerous mechanisms may still be
considered a Violation of <S1> or <G1> at the Head Referee’s discretion.
Violation Notes:
1. Most Violations of this rule should be considered Minor Violations. However, repeated
Minor Violations can escalate to a Major Violation at the Head Referee’s discretion,

---

26
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
especially in cases where Violations occur after multiple prior warnings.
2. Violations involving Scoring Objects Placed on the Midfield Goal that change the outcome
of the Match in favor of the offending Robot’s Alliance will be considered Match Affecting
Violations.
<SG13> Load Zones are protected during the Driver Controlled Period of the Match.
a. A Robot is protected while it is at least partially within one of its Alliance Load Zones. Robots may not
directly or indirectly contact a protected opposing Alliance Robot.
b. Robots may only enter an opposing Alliance Load Zone momentarily (e.g., while passing through or to
retrieve a Scoring Object); Robots may not remain within an opposing Alliance Load Zone.
c. Robots may not cause Scoring Objects to accumulate within an opposing Alliance Load Zone.
d. Robots may not add Scoring Objects to, or remove them from, opposing Alliance-colored Loaders.
The intent of this rule is to protect Load Zones from intentional interference by the opposing
Alliance (i.e., obstructing the Load Zone in an effort to inhibit Loader access).
This rule does not apply during the Autonomous Period.
Violation Notes:
1. Egregious Violations will be considered Major Violations.
2. Repeated Minor Violations can escalate to a Major Violation at the Head Referee’s discre-
tion, especially in cases where Violations occur after multiple prior warnings.

---

27
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Safety Rules
<S1> Be safe out there. If at any time the Robot operation or Team actions are deemed unsafe or have
damaged a Field Element, Scoring Object, or the Field, the offending Team may receive a Disablement and/or
Disqualification at the discretion of the Head Referee. The Robot will require re-inspection as described in rule
<R2> before it may take the Field again.
<S2> Students must be accompanied by an Adult. Every Student at a VEX Robotics Competition event must
be supervised by a responsible Adult. The Adult must obey all rules and be careful to not violate Student-cen-
tered policies, but must be present for the full duration of the event in the case of an emergency. Violations of
this rule may result in removal from the event and additional penalties.
<S3> Each Student Team member must have a completed participant release form on file for the event
and season. A Student Team member cannot participate in an event without a completed release form on file.
<S4> Stay inside the Field. If a Robot is completely outside of the Field during a Match, it will receive a Disable-
ment for the remainder of the Match.
Note: The intent of this rule is not to penalize Robots for having mechanisms that inadvertently cross
the Field Perimeter during normal game play.
<S5> Wear safety glasses. Drive Team Members must wear eye protection. All Drive Team Members must
wear some form of eye protection while at the Field for Matches. Safety glasses or other eye wear with side
shields and non-shattering lenses are recommended. While in the pit and queuing areas, it is highly recom-
mended that all Team members wear eye protection.

---

28
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# General Rules
<G1> Participants must follow the Code of Conduct. All event attendees must abide by the guidelines set
forth in the Code of Conduct.
Violation Notes:
1. Any Violation of <G1> may be considered a Major Violation and should be addressed on a
case-by-case basis. Teams at risk of a <G1> Major Violation due to multiple disrespectful
or uncivil behaviors will usually receive a “final warning”, although the Head Referee is not
required to provide one.
<G2> Participants must follow the Student Centered Policy. All event attendees must abide by the guide-
lines set forth in the Student Centered Policy.
Violation Notes:
1. Potential Violations of this rule will be reviewed on a case-by-case basis. By definition, all
Violations of this rule become Match Affecting as soon as a Robot which was built or pro-
grammed by an Adult scores points in a Match.
<G3> Use common sense. When reading and applying the various rules in this document, please remember
that common sense always applies in the VEX V5 Robotics Competition.
Some examples include:
● If there is an obvious typographical error (such as “per <T5>” instead of “per <GG5>”), this
does not mean that the error should be taken literally until corrected in a future update.
● Understand the realities of the VEX V5 Robot construction system. For example, if a Robot
could hover above the Field for a whole Match, that would create loopholes in many of the
rules. But… they can’t. So… don’t worry about it.
● When in doubt, if there is no rule prohibiting an action, it is generally legal. However, if you
have to ask whether a given action would violate <S1>, <G1>, or <T1> then that’s probably a
good indication that it is outside the spirit of the competition. On the other hand, if there’s
not a rule that makes a Robot part legal, it’s not allowed.
● In general, Teams will be given the “benefit of the doubt” in the case of accidental or edge-
case rules infractions. However, there is a limit to this allowance, and repeated or strategic
infractions will still be penalized.
● This rule also applies to Robot rules. If a component’s legality cannot be easily/intuitively
discerned by the Robot rules as written, then Teams should expect additional scrutiny
during inspection. This especially applies to those rules which govern non-VEX compo-
nents (e.g. <R18>, <R19>, <R22>, etc). There is a difference between “creativity” and “law-
yering.” Basically, if there’s not a rule that makes a Robot part legal, it’s not allowed.

---

29
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<G4> Students must meet the Student Eligibility Policy requirements. All Student participants must fall
within the eligibility requirements set forth in the Student Eligibility Policy for their respective program.
Violation Notes:
1. Teams believed to be in Violation of this rule should be reported to the Judge Advisor, Head
Referee, or Event Partner for further investigation in coordination with VEX Robotics. Based
on the investigation the Team may be removed from further Matches, have their Robot
Skills Match scores removed, and/or be removed from consideration from judged awards.
2. Violations of this rule will be evaluated on a case-by-case basis, in tandem with VEX
Robotics as noted in <G1> and <G2>.
<G5> There is a difference between accidentally and willfully violating a Robot rule. Any Violation of Robot
rules, accidental or intentional, will result in a Team being unable to play until they pass inspection (per <R2d>).
However, Teams who intentionally and/or knowingly circumvent or violate rules to gain an advantage over their
fellow competitors are in Violation of the spirit and ethos of the competition.
Violation Notes:
1. A Team that circumvents a Robot rule for a competitive advantage should receive an imme-
diate Disqualification for the current Match.

---

30
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<GG1> Only Drive Team Members, and only in the Alliance Station. During a Match, Robots may be operated
only by that Team’s Drive Team Members and/or by software running on the Robot’s control system in accor-
dance with <R9> and <GG11>. A Team may send up to (3) Drive Team Members to their Alliance Station for each
Robot, and those Drive Team Members must remain in their Alliance Station for the duration of the Match.
Drive Team Members are the only Team members that are allowed to be in the Alliance Station during a Match.
Adults (other than event staff) are not permitted to be in the Alliance Station during a Match.
a. Drive Team Members are prohibited from any of the following actions during a Match:
i. Using any sort of communication device in the Alliance Station. Non-headphone devices with commu-
nication features turned off (e.g., a phone in airplane mode, a walkie talkie turned off, smart glasses with
communication features disabled) are allowed. Communication features can be enabled for translation
apps during post-Match discussions.
ii. Standing or sitting on any sort of object during a Match, regardless of whether the Field is on the floor
or elevated, except as required by an officially approved accommodation request.
iii. Bringing/using additional materials to simplify the game challenge during a Match (e.g., device to align
or add Scoring Objects to a Loader.
iv. To ensure that Drive Team Members are aware of verbal calls during a Match (as an application of rules
<T1>, <G1>, <S1>, and <G3>), powered headphones, earbuds, passive earpieces connected to elec-
tronic devices, or other personal accessories/devices that transmit audio cannot be worn/used in the
Alliance Station except as required by an officially approved accommodation request.
b. Individuals who are not Drive Team Members for a Match cannot provide directions, commands, or advice
to the Drive Team Members during that Match. They’re welcome to provide cheerful, positive encourage-
ment, but should not affect Match play or strategy.
Point iii is intended to refer to non-Robot-related items that directly influence gameplay, such
as a speaker that plays a buzzer sound to distract your opponent. Provided no other rules are
violated, and the items do not pose any safety or Field damage risks, the following examples
are not considered Violations of <GG1>:
● Materials used before or after a Match, such as a pre-Match alignment aid
● Strategic aids, such as a whiteboard or clipboard
● Earplugs, gloves, or other personal accessories
Violation Notes:
1. Major Violations of this rule are not required to be Match Affecting, and could invoke Viola-
tions of other rules, such as <G1>, <G2>, or <G4>.
<GG2> A Team’s Robot should attend every Match. The Team’s Robot must be in the Alliance Station or
on the Field for the Team’s assigned Match, even if the Robot is not functional. If the Robot is not at the Field
for the entire duration of the Match, the Team will be considered a “no-show” and receive zero (0) Win Points,
Autonomous Win Points, Autonomous Points, and Strength of Schedule Points.
# General Game Rules

---

31
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
a. Teams are expected to participate in all scheduled Qualification Matches, Alliance Selection, and Elimina-
tion Matches (if they’re an Alliance Captain or were selected to join an Alliance for Elimination Matches).
Failure to attend scheduled Matches or Alliance Selection may be considered a Violation of <G1>. Teams
that participate in zero Qualification Matches cannot be considered for judged awards.
<GG3> Robots on the Field must be ready to play. When a Team puts their Robot on the Field, it must be
prepared to play (e.g., battery charged, sized within the starting size constraint, includes only the correct
Alliance-color license plates, etc.).
a. Robots must be placed on the Field promptly. Repeated failure to do so could result in a Violation of <G1>
and/or removal of the Robot from the current Match at the Head Referee’s discretion.
b. If a Robot is delaying the scheduled start of a Match, it may be removed from the Field at the discretion
of the Head Referee and Event Partner. The Robot may remain at the Field so that the Team does not get
assessed a “no-show” (per <GG2>).
c. If a Robot is not placed on the Field prior to the start of a Match, it cannot be placed on the Field during that
Match.
d. Teams who use VEX pneumatics must have their systems charged before they place the Robot on the Field.
e. If an event is using Smart Field Control and a Robot is unable to successfully connect to Smart Field Control
prior to the scheduled start of a Match, the Head Referee may ask the Team to remove their Robot from the
Field in accordance with clause B.
i. A Robot that connects to Smart Field Control but displays a ‘Legacy Field Control’ error on the field
monitor is NOT considered successfully connected to Smart Field Control, and may be removed from
the Field if it is delaying the scheduled start of a Match.
The definition of the word “promptly” as used in clause A is at the discretion of the Event
Partner and Head Referee, who will consider event schedule, previous Violations or delays, etc.
As a general guideline, five seconds to check Robot alignment would be acceptable, but five
minutes to assemble multiple parts together would not.
<GG4> Hands out of the Field. During a Match, Drive Team Members are prohibited from reaching into the
3-dimensional volume of the Field Perimeter and from making intentional contact with any Field Element,
Robot, or Scoring Object that has been introduced to the Field, unless otherwise noted below.
a. Drive Team Members are allowed to reach into the Field during the Match while introducing Match Loads as
described in rule <SG11>. Rule <S1> applies.
b. Any concerns regarding Field Element or Scoring Object starting positions should be raised with the
Head Referee prior to the Match. Team members may never adjust Scoring Objects or Field Elements
themselves.
c. Transitive contact, such as contact with the Field Perimeter that causes the Field Perimeter to contact Field
Elements or objects inside of the Field, could be considered a Violation of this rule.
d. During the Driver Controlled Period, Drive Team Members may reach into the Field and touch their own
Robot only if the Robot has not moved at all during the Match. Movement caused by an external force, such
as another Robot, should not prevent a Drive Team Member from interacting with their Robot under this
rule. Touching the Robot in this case is permitted only for the following reasons:

---

32
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
i. Turning the Robot on or off.
ii. Plugging in a battery.
iii. Plugging in a V5 Robot Radio.
iv. Touching the V5 Robot Brain screen, such as to start a program.
If a Drive Team Member’s hands extend over the Field and/or Field Perimeter in a way that is
safe and doesn’t contact anything in the Field, it’s unlikely to be a Violation. However, Head
Referees may still ask Drive Team Members to step back and remain completely outside
the Field when necessary (e.g., for safety reasons or to reduce the chances of gameplay
interference).
<GG5> Match replays are allowed, but rare. Match replays (i.e., playing a Match over again from its start)
must be agreed upon by both the Event Partner and Head Referee, and will only be issued in the most extreme
circumstances. Some example situations that may warrant a Match replay are as follows (note that this is not
an exhaustive list):
a. Match Affecting “Field fault” issues.
i. One or more Field Elements and/or Scoring Objects starting in incorrect positions, and out of the
allowed tolerances (see <T5>).
ii. Tape lines lifting.
iii. Field Elements detaching or moving beyond normal tolerances (not as a result of Robot interactions).
iv. The Autonomous Period or Driver Controlled Period ending early.
v. Field control disconnecting or Disabling multiple Robots. Note, this is sometimes confused with a Robot
whose motors have overheated, or bent pins on a controller’s competition port causing intermittent
drop-outs. In general, any true Field fault will impact both Alliances simultaneously, not one Robot at a
time.
b. Match Affecting game rule issues.
i. Head Referee Disables a Robot for a misinterpretation of a rule Violation.
ii. Head Referee starts the Driver Controlled Period of the Match without determining the outcome of the
Autonomous Period winner.
c. The Field is reset before a score is determined.
d. A Match is run before its scheduled time without a Team.
Note: Communication or control issues affecting a single Robot are not considered eligible
grounds for a replay.
<GG6> Disqualifications. When a Team receives a Disqualification in a Qualification Match, they receive a
score of zero (0) for the Match, as well as zero (0) Win Points, Autonomous Win Points, Autonomous Points, and
Strength of Schedule Points.
a. If the Team receiving the Disqualification is on the winning Alliance, then Teams on the opposing Alliance
who are not also Disqualified will receive the win for the Match and two (2) Win Points.

---

33
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
i. The Team’s non-Disqualified Alliance Partner is unaffected, i.e., they will also receive the win for the
Match and two (2) Win Points.
b. If the Match was a tie, then each Team on the opposing Alliance (the Alliance that did not receive the Disqual-
ification) will receive the win for the Match and two (2) Win Points. If both Alliances have a Team receiving a
Disqualification, then all non-Disqualified Teams will receive a tie for the Match and one (1) Win Point.
c. Autonomous Win Points are not given to Teams that receive a Disqualification, and are not automatically
awarded to the opposing Alliance.
When a Team is Disqualified in an Elimination Match, the entire Alliance is Disqualified; they receive a loss for the
Match, and the opposing Alliance is awarded the win. If both Alliances receive a Disqualification in an Elimination
Match, both Alliances receive a loss and will play another Match to determine a winner.
Note: If a Team is Disqualified in a Robot Skills Match, a score of zero (0) will be recorded for that Match.
<GG7> Time-outs. Each Elimination Alliance gets one three-minute Time-out, which they may request during
the Elimination Bracket. The Time-out will be served at the time of the Alliance’s next upcoming Match. Alliances
must request their Time-out between Elimination Matches; they may not use their Time-out during a Match, for
another Alliance’s Match, or after they have been eliminated. There are no Time-outs during the Qualification
Match schedule.
a. A Time-out can be ended early, but only if agreed to by both Alliances and the Head Referee.
b. An Alliance’s Time-out request should never be denied if the Alliance legitimately needs extra time.
<GG8> Keep your Robots together. Robots may not intentionally detach parts during the Match or leave mech-
anisms on the Field.
Note: Parts which become detached unintentionally are a Minor Violation, are no longer considered “part
of a Robot,” and should be ignored for the purpose of any rules which involve Robot contact or location
(e.g., scoring) or Robot size.
Violation Notes:
1. Major Violations of this rule should be rare, as Robots should never be designed to intention-
ally violate it. Minor Violations are usually due to Robots being damaged during gameplay,
such as a wheel falling off.
<GG9> Don’t hook your Robot to the Field, and don’t get Entangled. Robots may not intentionally grasp,
grapple, hook, attach to or otherwise Entangled with any Field Elements. Strategies with mechanisms that react
against multiple sides of a Field Element in an effort to latch or hook onto said Field Element are prohibited. The
intent of this rule is to prevent Teams from unintentionally damaging the Field and/or from anchoring to or other-
wise Entangling themselves with the Field.
Whenever possible, Head Referees should alert Teams to potential Violations before they happen to prevent
actual Violations. If a Robot takes immediate action to avoid or resolve the issue, and if the Head Referee deter-
mines that the issue had no effect on the Match, no Violation should be recorded.

---

34
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<GG10> The red Alliance may choose to place last. The red Alliance has the right to place its Robots on the
Field last in Qualification Matches and Elimination Matches. Once a Team has placed its Robot on the Field, in
order to avoid schedule delays its position should not be adjusted prior to the Match. <GG3> applies. If a Team
chooses to reposition their Robot after it has already been placed, the opposing Alliance will also be given the
opportunity to reposition their Robots promptly.
This rule is applied differently for VEX U. See Rule <VUG2>
<GG11> Controllers must stay connected to the Field. Prior to the beginning of each Match, Drive Team
Members must plug their V5 Controller into the Field’s control system. This cable must remain plugged in for
the duration of the Match, and may not be removed until the “all-clear” has been given for Drive Team Members
to retrieve their Robots. See <T8> for more information regarding Field control system options.
Violation Notes:
1. The intent of this rule is to ensure that Robots abide by commands sent by the tournament
software. Temporarily removing the cable to assist with mid-Match troubleshooting, with an
Event Partner or other event technical staff present and assisting, would not be considered
a Violation.
<GG12> Autonomous means “no humans.” During the Autonomous Period, Drive Team Members are not
permitted to interact with the Robots in any way, directly or indirectly. This could include, but is not limited to:
● Activating any controls on their V5 Controllers
● Unplugging or otherwise manually interfering with the Field connection in any way
● Manually triggering sensors (including the Vision Sensor) in any way, even without touching them
Note: In extreme cases, with permission from the Head Referee, Teams may Disable their Robot during
the Autonomous Period by holding the power button on their V5 Controller. This exception is only
intended for egregious safety- or damage-related circumstances; Disabling an autonomous routine for
strategic purposes would still be considered a Violation of <GG12>.
Violation Notes:
1. See <GG13>.
<GG13> All rules still apply in the Autonomous Period. Teams are responsible for the actions of their Robots
at all times, including during the Autonomous Period. Any Violations, Major or Minor, committed during the
Autonomous Period will result in the Autonomous Bonus being automatically awarded to the opposing Alliance
and make the violating Team’s Alliance ineligible for the Autonomous Win Point.
If both Alliances commit Violations during the Autonomous Period, then no Autonomous Bonus will be awarded.

---

35
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Violation Notes:
1. In general, Minor Violations of SG rules that occur during the Autonomous Period should
only affect the outcome of the Autonomous Period (i.e., the Alliance can’t win the Autono-
mous Bonus or earn an Autonomous Win Point) and should not be considered when deter-
mining whether a Violation has been repeated during the event.
2. If a Head Referee determines that a Violation of an SG or GG rule during the Autonomous
Period was intentional/strategic rather than accidental/situational, it should be recorded as
a Minor Violation or Major Violation and considered when determining whether a Violation
has been repeated during the event.
<GG14> Don’t destroy other Robots. But, be prepared to encounter defense. Strategies aimed solely at the
destruction, damage, tipping over, or Entanglement of opposing Robots are not part of the ethos of the VEX V5
Robotics Competition and are not allowed.
a. V5RC Override is intended to be an Offensive game. Teams that partake in solely Defensive or destructive
strategies will not have the protections implied by this rule (see <GG15>). However, Defensive play which
does not involve destructive or illegal strategies is still within the spirit of this rule.
b. V5RC Override is also intended to be an interactive game. Some incidental tipping, Entanglement, and
damage may occur as a part of normal gameplay without Violation. It will be up to the Head Referee’s dis-
cretion whether the interaction was incidental or intentional.
c. A Team is responsible for the actions of its Robot at all times, including the Autonomous Period. This applies
both to Teams that are driving recklessly or potentially causing damage, and to Teams whose Robots have
small and/or unstable wheel bases. A Team should design its Robot such that it is not easily tipped over or
damaged by minor contact.
d. <GG14> may be applied differently in the Midfield during the Endgame. See <SG12>.
Violation Notes:
1. Major Violations of this rule are not required to be Match Affecting. Intentional and/or egre-
gious tipping, Entanglement, or damage may be considered a Major Violation at the Head
Referee’s discretion.
2. Repeated Violations within a Match or Tournament could be considered a Violation of <G1>
and/or <S1> at the Head Referee’s discretion.
<GG15> Offensive Robots get the “benefit of the doubt” when judgment calls are required. In a case where
a Head Referee is forced to make a judgment call regarding a destructive interaction between a Defensive and
Offensive Robot, or an interaction which results in a questionable Violation, referees will decide in favor of the
Offensive Robot. This also applies during the Autonomous Period (see <SG7a>).
This rule is intended to be a “logical tiebreaker” for use only when referees cannot make a clear
and definitive ruling without it. It should not be used or applied in every situation. Most Viola-
tions should be resolved without considering this rule.

---

36
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Head Referees must apply judgment when determining whether each Robot in a <GG15>
interaction was Defensive or Offensive, and in some cases may need to consider which Robot
was more Defensive or Offensive than another within the larger context of the Match. In these
cases, the Head Referee should decide in favor of the less Defensive and/or more Offensive
Robot based on the definitions and guidance in this game manual.
<GG16> You can’t force an opponent into a penalty. Actions that cause an opposing Robot to break a rule are
not permitted, and will not result in a Violation for the opposing Robot.
Violation Notes:
1. In most cases, if a Team causes their opponent to break a rule, the Head Referee will simply
not enforce the penalty on that opponent, and it will be considered a Minor Violation for
the Team that forced a Violation. However, if the forced situation becomes Match Affecting
in favor of the Team that forced the Violation, it will be considered a Major Violation for the
Team that forced the Violation.
<GG17> No Holding for more than a 3-count. A Robot may not Hold an opposing Robot for more than a
3-count during the Driver Controlled Period.
For the purposes of this rule, a “count” is defined as an interval of time that is approximately one second in
duration, and “counted out” by Head Referees verbally. A Holding count should begin immediately once the
Head Referee observes a suspected Holding interaction.
The Holding count should pause when at least one of the following conditions is met:
a. The two Robots are separated by at least two (2) feet (approximately one foam tile).
b. Either Robot has moved at least two (2) feet away (approximately one tile) from the location where the
Trapping or Pinning count began.
i. In the case of Lifting, this location is measured from where the Lifted Robot is released, not from where
the Lifting began.
c. The Holding Robot becomes Trapped or Pinned by a different Robot.
i. In this case, the original count would end, and a new count would begin for the newly Trapped or Pinned
Robot.
d. In the case of Trapping, if an avenue of escape becomes available due to changing circumstances in the
Match.
After a Holding count ends, a Robot may not resume Holding the same Robot again for a 5-count. If a Team
resumes Holding the same Robot within that 5-count, the original Holding count will resume from where it
ended. A Head Referee should use fingers to display the 5-count that occurs after the end of a Holding count,
and “wave it off” after the Holding interaction has been cleared.
If two Robots are working together to Trap an opponent simultaneously, the Holding count can be applied to
both Robots; it’s possible for them to legally take turns Trapping, but it’s risky.

---

VEX V5 Robotics Competition Override - Game Manual
37
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
If the Head Referee determines that a Robot is not attempting to escape, then it is not considered
Pinned or Trapped. This commonly occurs when the Robot has malfunctioned and lost the ability
to move, or when the Robot is defending a Field Element.
This criteria is not required for Lifting; the Holding status begins as soon as the opponent
becomes Lifted.
Holding is a standard and legal part of Head-to-Head game play, and only becomes a Violation
if it exceeds the guidelines in this rule. By beginning a Holding count immediately after noticing
a Holding interaction, and providing a visual signal when a Holding interaction has been cleared,
Head Referees can help Teams avoid penalties.
Starting a Holding count is not, in and of itself, a declaration that Holding is occurring. Ending a
Holding count early and waving it off is also not, in and of itself, an indication that Holding has
occurred. There is no harm in a Head Referee reacting quickly with a Holding count, realizing that
no Holding is occurring, and then ending and waving off the count.
During a Holding count, the Head Referee should continually verify that the interaction meets
the definition of Holding. If it becomes clear that the interaction doesn’t meet the definition of
Holding, the Head Referee should end the count early and wave off the Holding count. Holding only
becomes a Violation if a referee exceeds a three-count before the Robots separate as described
in rule <GG17>.
Violation Notes:
1. When determining whether a Holding Violation was Match Affecting, they should consider the
full context of the Match and both Robots’ actions, including the following. Because the first
three seconds of a Hold are legal game play and not a Violation, only the extended portion of a
Hold should be considered. Holding is an inherently Defensive action, so rule <GG15> may also
be considered if a judgment call is required.
० How much extra time was included in the Holding interaction
० How long the Robots were separated if the Holding Robot backs away and returns too early
० What both Robots were doing within the larger context of the Match
<GG18> Use Scoring Objects to play the game. Scoring Objects may not be used to accomplish actions that
would be otherwise illegal if they were attempted by Robot mechanisms. If a rule is Violated through the use of a
Scoring Objects instead of a Robot mechanism, it should be evaluated as though the rule in question had been
Violated by a Robot mechanism. Examples include, but are not limited to:
● Interfering with an opponent’s Autonomous routine per <SG7>
● Using a Scoring Objects to intentionally tip or Entangle an opponent Robot
The intent of this rule is to prohibit Teams from using Scoring Objects as “gloves” to loophole any rule that states
“a Robot may not [do some action].” This rule is not intended to be taken in its most extreme literal interpretation,
where any interaction between a Scoring Object and a Robot needs to be scrutinized with the same intensity as if it
were a Robot.

---

Copyright 2026, VEX Robotics, Inc.
## Section 3
## Robot Skills

---

39
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Overview
# Section 3 - Robot Skills
In Robot Skills Matches, Teams have one minute to score as many points as possible. There are two types of
Robot Skills Matches: Driving Skills Matches, which are entirely Driver controlled, and Autonomous Coding
Skills Matches, which are autonomous with limited human interaction. Teams are ranked based on their
combined score in the two types of Robot Skills Matches.
Robot Skills Matches are optional for all Teams. Teams who do not compete in Robot Skills Matches will not
be penalized in Teamwork Matches. However, participation in Robot Skills Matches may impact eligibility for
judged awards at the event.
At events that include Teamwork Matches, Teams may only participate in Robot Skills Matches if they also
participate in the Qualification Matches. See rule <T20>.
# Robot Skills Challenge Rules
<RSC1> Standard rules apply in most cases. All rules from previous sections apply to Robot Skills Matches,
unless otherwise specified in this section.
a. Removing Scoring Objects from the Field in a Robot Skills Match is not a Violation. Scoring Objects that
leave the Field cannot be returned.
b. In the Robot Skills Challenge, the standard definition of Match Affecting does not apply, because there is no
winner or loser. When evaluating whether a rule Violation should be classified as a Major Violation or Minor
Violation in the context of this criteria, the term “score affecting” can be substituted for “Match Affecting.” A
Violation is considered “score affecting” if it results in a net increase of that Team’s score at the end of the
Match.
c. Violations of <GG>, <SG>, and <RSC> rules that occur during a Robot Skills Match should only affect the
outcome of that Match and should not be considered when determining whether a Violation has been
repeated during the event.
<RSC2> Match play is different in Robot Skills Matches.
a. The Robot must start the Robot Skills Match in a legal starting position in the Quadrant adjacent to the red
Alliance Station.
b. All Drive Team Members must remain in the red Alliance Station for the duration of the Match.
c. One red/yellow Pin must be used as a Preload in accordance with <SG5>.
d. Teams may introduce Match Load Scoring Objects at any time during a Robot Skills Match, but only through
red Alliance Loaders.
e. Robots can add or remove Pins or Cups on all Goals.
f. Robots may move freely about the Field after the start of the Match.
This rule is applied differently for VEX U. See Rule <VURS3>.
<RSC3> Scoring Robot Skills Matches. For each Robot Skills Match, Teams are awarded a score based on the
following rules and scoring table:

---

40
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
a. Teams can earn points for Pins of any color.
i. Red Pins can only be Scored if they are Placed in a red Quadrant or in the Midfield.
ii. Blue Pins can only be Scored if they are Placed in a blue Quadrant or in the Midfield.
iii. Yellow Pins can only be Scored if they are Owned.
1. Toggles only confer Ownership of Pins when the Toggle matches the color of its Quadrant (i.e.,
yellow Pins Placed in a red Quadrant with a Toggle that is set to blue are not Owned and will not earn
any points.
2. Yellow Pins Placed in the Midfield are Owned if a Robot ends the Match in the Midfield. (See <SC6>).
b. The Team will earn points for a Robot ending in the Midfield. (See <SC6>.)
Each Scored red or blue Pin 5 points
Each Scored yellow Pin 10 points
Each Robot ending in the Midfield 8 points
<RSC4> Field setup for Skills Matches. The Field is set up differently than a Head-to-Head Match, with the
following modifications:
a. In Autonomous Coding Skills Matches, the VEX GPS code strip must be installed on the Field.
b. Revised Match Loads. Three (3) red/yellow Pins, four (4) blue/yellow Pins, and seven (7) Cups begin off the
Field as Match Loads.
c. All Goals start the Match empty with no Placed Pins.
This rule is applied differently for VEX U. See Rule <VURS1>.
Figure RSC4-1: An overhead
view of a V5RC Override
Robot Skills Match.

---

41
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<RSC5> Skills Stop Time. If a Team wishes to end their Robot Skills Match early, they may elect to record a
Skills Stop Time. This is used as a tiebreaker for Robot Skills Challenge rankings. A Skills Stop Time does not
affect a Team’s score for a given Robot Skills Match. Drive Team Members and field staff must agree prior to
the Match on the signal that will be used to end the Match early.
a. Teams who intend to attempt a Skills Stop Time must “opt-in” by verbally confirming with the Scorekeeper
Referee prior to the Robot Skills Match. If no notification is given prior to the start of the Match, then the
Team forfeits their option to record a Skills Stop Time for that Match and they will receive a default Skills
Stop Time of 0.
i. This conversation should include informing the Scorekeeper Referee which Drive Team Member will
signal the stop. The Match may only be ended early by a Drive Team Member for that Match.
ii. If a Team runs multiple Robot Skills Matches in a row, they must reconfirm their Skills Stop Time choice
with the Scorekeeper Referee prior to each Match.
iii. Any questions regarding a Skills Stop Time should be reviewed and settled immediately following the
Match. <T1> and <T3> apply to Robot Skills Matches.
b. If the event is utilizing a V5 Robot Brain or the TM Mobile app for Robot Skills Challenge field control, a Drive
Team Member may elect to start and stop their own Robot Skills Matches.
i. This V5 Robot Brain or other device running the TM Mobile app will be used to start the Robot Skills
Matches (i.e., “enable” the Robot), end the Robot Skills Match (i.e., “Disable” the Robot), and display the
official Skills Stop Time to be recorded.
ii. This V5 Robot Brain must be running the official field control user program.
iii. For more information regarding the use of a V5 Robot Brain for Robot Skills Challenge field control, and
to download the official field control user program, visit this VEX Knowledge Base article.
iv. For more information regarding the use of TM Mobile for field control, see the Tournament Manager
documentation.
c. At events which do not have a V5 Robot Brain or the TM Mobile App available for Robot Skills Challenge field
control, Drive Team Members and field staff must agree prior to the Match on the signal that will be used to
end the Match early. A manual timer must be used in conjunction with a VEXnet Competition Switch.
i. The moment when the Match ends early is defined as the moment when the Robot is “Disabled” by the
field control system. The time shown on the timer should be rounded up to the nearest second. For
example, if the Robot is Disabled and the timer shows 25.2 seconds, then the Skills Stop Time should be
recorded as 26.
ii. The agreed-upon signal must be both verbal and visual, such as Drive Team Members crossing their
arms in an “X” or placing their V5 Controller(s) on the ground.
iii. The signal must be given by a Drive Team Member who is standing in the Alliance Station.
iv. It is recommended that Drive Team Members also provide verbal notice that they are approaching their
Skills Stop Time, such as by counting out “3-2-1-stop.”
d. It is at the Event Partner’s discretion which method will be used to record Skills Stop Times at a given event.
The chosen method must be communicated prior to the start of Matches (such as during an event meeting),
and made equally available to all Teams.
i. If an event intends to use a manual timekeeping method, a Team cannot bring their own V5 Robot Brain
just for use during their own Robot Skills Match.
ii. If an event intends to utilize a V5 Robot Brain, all Teams must use the same V5 Robot Brain for all Robot
Skills Matches on a given Field.

---

VEX V5 Robotics Competition Override - Game Manual
42
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
iii. If an event is using multiple Fields for Robot Skills Matches, the same method must be used at all Fields,
as described in rule <T21>. Multiple V5 Robot Brains may be used as needed (e.g., a “Field 1 Brain” and a
“Field 2 Brain”).
iv. The default timed “Drive” program accessed from a V5 Controller is intended for practice only, and
cannot be used for an official Robot Skills Match.
e. If a Team chooses to utilize/record a Skills Stop Time, the 5-second grace period described in rule <SC1>
does not apply.

---

Copyright 2026, VEX Robotics, Inc.
## Section 4
## The Robot

---

44
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Overview
# Section 4 - The Robot
# Robot Rules
All Robots must pass inspection before competing in the VEX V5 Robotics Competition to ensure compliance
with all Robot rules. Inspections typically occur during Team check-in or practice time. Teams should use the
rules below to pre-inspect their Robot and confirm it meets all requirements.
Most rules are hard limits (e.g., motor quantity), while others are subject to inspector discretion (e.g., safety
concerns). At many events, the lead inspector and Head Referee are the same person; if not, inspectors should
consult the Head Referee for judgment calls. Per <R2d> and <R2e>, the Head Referee has final authority on all
Robot rules and whether a Robot may compete.
<R1> One Robot per Team. Each Team can only bring one Robot to a given event in the VEX V5 Robotics
Competition. Though it is expected that Teams will make changes to their Robot at the competition, a Team is
limited to only one Robot at a given event, and a given Robot may only be used by one Team. A VEX Robot, for
the purposes of the VEX V5 Robotics Competition, has the following subsystems:
● Subsystem 1: Mobile robotic base including wheels, tracks, legs, or any other mechanism that allows the
Robot to navigate the majority of the flat playing Field surface. For a stationary Robot, the robotic base
without wheels would be considered Subsystem 1.
● Subsystem 2: Power and control system that includes a legal VEX battery, a legal VEX control system, and
associated motors for the mobile robotic base.
● Subsystem 3: Additional mechanisms (and associated motors) that allow manipulation of Scoring Objects
and interactions with Field Elements and other Robots.
Given the above definitions, a minimum Robot for use in any VEX V5 Robotics Competition event (including
Robot Skills Matches) must consist of subsystems 1 and 2 above. Thus, if you swap out an entire subsystem 1
or 2, you have created a second Robot and are in Violation this rule.
a. Teams may not compete with one Robot while a second is being modified or assembled at a competition.
b. Teams may not have an assembled second Robot on hand at a competition that is used to repair or swap
parts with the first Robot.
c. Teams may not switch back and forth between multiple Robots during a competition. This includes using
different Robots for Robot Skills Matches, Qualification Matches, and/or Elimination Matches.
d. Multiple Teams may not use the same Robot. Once a Robot has competed under a given Team number at an
event, it is “their” Robot; no other Team may EVER compete with it.
The intent of <R1a>, <R1b>, and <R1c> is to ensure an unambiguous level playing field for all
Teams. Teams are welcome (and encouraged) to improve or modify their Robots between
events, or to collaborate with other Teams to develop the best possible game solution.
However, a Team who brings and/or competes with two separate Robots at the same Tour-
nament has diminished the efforts of a Team who spent extra design time making sure that
their one Robot can accomplish all of the game’s tasks. A multi-Team organization that shares
a single Robot has diminished the efforts of a multi-Team organization who puts in the time,

---

45
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
effort, and resources to undergo separate individual design processes and develop their own
Robots.
To help determine if a Robot is a “separate Robot” or not, use the subsystem definitions found
in <R1>. Above that, use common sense as referenced in <G3>. If you can place two Robots
on a table next to each other, and they look like two separate legal/complete Robots (i.e., each
has the 3 subsystems defined by <R1>), then they are two Robots. Trying to decide if changing
a screw, a wheel, or a microcontroller constitutes a separate Robot is missing the intent and
spirit of this rule. Rules <G4> and <R2> still apply to both Robots.
This rule is applied differently for VEX U. See Rule <VUR1>.
<R2> Robots must pass inspection. Every Robot will be required to pass a full inspection before being
cleared to compete. This inspection will ensure that all Robot rules and regulations are met. Noncompliance
with any Robot design or construction rule will result in removal from Matches or Disqualification of the Robot
at an event until the Robot is brought back into compliance, as described in the following subclauses.
a. Significant changes to a Robot, such as a partial or full swap of Subsystem 3, must be re-inspected before
the Robot may compete again.
b. All possible functional Robot configurations must be inspected before being used in competition. This
especially pertains to modular or swappable mechanisms (per <R1>) and Match starting configurations/
sizes (per <R3>).
c. Teams may be requested to submit to spot inspections by Head Referees. Refusal to submit will result in
Disqualification.
i. If a Robot is determined to be in Violation of a Robot rule before a Match begins, the Robot will be
removed from the Field. The Robot may remain at the Field so that the Team does not get assessed a
“no-show” (per <GG2>).
d. Robots which have not passed inspection (i.e., that are in Violation of one or more Robot rules) will not be
permitted to play in any Matches until they have passed inspection. <GG2> will apply to any Matches that
occur until the Robot has passed inspection.
e. If a Robot has passed inspection, but is later confirmed to be in Violation of a Robot rule during or immedi-
ately following a Match by a Head Referee, they will be Disqualified from that Match. This is the only Match
that will be affected; any prior Matches that have already been completed will not be revisited. <R3d> will
apply until the Violation is remedied and the Team is re-inspected.
f. All robot rules are to be enforced within the discretion of the Head Referee within a given event. Robot
legality at one event does not automatically imply legality at future events. Robots which rely on “edge-
case” interpretations of subjective rules, such as whether a decoration is “non-functional” or not, should
expect additional scrutiny during inspection.
g. Events may wish to use “inspection markers” (e.g., zip tie or sticker) to identify Robots that have passed
inspection at that event. Inspection markers are functional components and are subject to all Robot rules,
including legal materials and Robot size/expansion limits.
Event staff and volunteers are allowed to contact and/or photograph Robots during inspection and/or at other
times as needed.

---

46
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<R3> Robots must fit within an 18” x 18” x 18” volume.
a. Compliance with this rule may be checked using the official VEX Robotics On-Field Robot Expansion Sizing
Tool.
b. Event Partners may construct and/or provide any sizing tool that measures the correct dimensions.
c. Any restraints used to maintain starting size (i.e., zip ties, rubber bands, etc.) must remain attached to the
Robot for the duration of the Match, per <GG8>.
d. For the purposes of this rule, it can be assumed that Robots will be inspected and begin each Match on a
flat standard foam field tile.
The official sizing tool is intentionally manufactured with a slightly oversized tolerance. There-
fore, any contact with the sizing tool (i.e., a “paper test”) while being measured should be
considered a clear indication that a Robot is outside of the permitted size. This tolerance also
provides a slight “leeway” for minor protrusions, such as screw heads or zip ties.
This rule is applied differently for VEX U. See Rule <VUR1>.
<R4> Officially registered Team numbers must be displayed on Robot license plates. To participate in an
official VEX V5 Robotics Competition event, a Team must first register on events.vex.com and receive a V5RC
Team number. This Team number must be displayed on the Robot using license plates. Teams may choose to
use the official V5RC License Plate Kit, or may create their own using legal materials.
a. License plates must be placed in fixed locations on exactly two (2) horizontally opposing sides of the Robot
and must remain visible, legible, and attached for the entirety of the Match. The top of a Robot is not con-
sidered a “side” for these two license plates.
i. License plates should be mounted in locations that remain stationary on the Robot during a Match (e.g.,
not on a rotating intake or flipping manipulator). The function of license plates is to identify Robots for
referees, spectators, and other Teams. Identification is harder when a license plate on a Robot moves
during a Match.
b. License plates must be securely attached to the Robot using materials that are legal for Robot construc-
tion. VEX IQ pins are not legal for mounting license plates on Robots.
c. Robots may only include license plates that match their Alliance color for the current Match (i.e., red
Alliance Robots must have only red plates installed for the Match).
d. License plates are considered functional components, and must meet the requirements of all Robot rules.
e. Additional license plates cannot be used on the Robot for any purpose.
f. Team numbers must be in white font, and clearly legible. Number/letter stickers are legal for use on license
plates instead of or in addition to those in the V5RC License Plate Kit.
g. Custom license plates used to meet the requirements of <R4a> and <R4f> must be within the following size
limits:
i. Height: between 2.0 (50.8mm) and 2.5 inches (63.5mm)
ii. Width: between 4.0 (101.6mm) and 4.5 inches (114.3mm)
iii. Thickness: up to 0.25” (6.35mm)

---

47
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Imagine the Robot as a cube that’s sitting flat on a flat surface. License plates should be placed
onto two (2) opposing faces of that cube (excluding the top and bottom).
When a license plate is mounted to the Robot, only human influence should be able to
remove the plate. The mounting should be secure enough to withstand any and all interac-
tions between the Robot and other things on the Field. Using rubber bands or other flexible
mounting methods to mount license plates should not be considered secure, and should not
pass inspection. Using screws and nuts/standoffs to mount plates is the best way to meet the
requirements for securely mounted license plates.
The intent of this rule is to make it immediately apparent to Head Referees and other event
personnel which Alliance and which Team each Robot belongs to, at all times. It will be at the
full discretion of the Head Referee and inspector at a given event to determine whether a given
license plate satisfies the criteria listed in <R4>.
Teams who use custom plates should be especially prepared for the possibility of this
judgment, and ensure that they are prepared to replace any custom parts with official VEX
license plates if requested. Not bringing official replacement plates to an event will not be an
acceptable reason for overlooking a Violation of one or more points in <R4>. Teams are en-
couraged to use an easily-read, sans-serif font (e.g., Arial).
If a Robot must be removed from the Field based on this rule, <R2ci> applies and the Team
should not be issued a “no-show.”
Figure R4-1: An example of a license plate made from the V5RC License
Plate Kit.
Figure R4-2: An example of a legal custom license plate
<R5> Let go of Scoring Objects after the Match. Robots must be designed to permit easy removal of Scoring
Objects from any mechanism without requiring the Robot to have power after a Match.
<R6> Robots have one Brain. Robots must ONLY use one (1) VEX V5 Robot Brain (276-4810). Any other micro-
controllers or processing devices are not allowed, even as non-functional decorations.
a. This includes microcontrollers that are part of other VEX product lines, such as VEX Cortex, VEX EXP,
VEXpro, VEX CTE, VEX RCR, VEX IQ, VEX GO, VEX AIR, or VEX Robotics by HEXBUG. This also includes
devices that are unrelated to VEX, such as Raspberry Pi or Arduino devices.
b. V5 Robot Brain accessories (short flanges, long flanges, and the magnetic screen protector) are part of the
V5 Robot Brain and are only legal for use on Robots as part of the V5 Robot Brain.

---

48
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
c. At events using a V5 Smart Field Control System, the “Team Number” field in the Robot Brain must be set as
the Team’s registered number and letter (with no spaces).
This rule is applied differently for VEX U. See Rules <VUR10> and <VUR12>.
<R7> Keep the power button or battery connection accessible. The on/off button on the V5 Robot Brain and/
or the Battery Cable connection on either the V5 Robot Brain or V5 Robot Battery must be accessible without
moving or lifting the Robot. The V5 Brain screen should be easily visible during Robot inspection. Keeping the
V5 Brain screen visible throughout a Match is recommended but not required.
This rule is in place to ensure the safety of both competitors and field staff. In the event that
a Robot needs to be quickly powered off—whether due to a malfunction, Entanglement, or
other safety concern—it is crucial that the power button and/or Robot Battery remains easily
accessible. This allows competitors and/or field personnel to safely Disable the Robot without
putting their hands near moving parts or other hazards inside the Robot.
Additionally, keeping screens and indicator lights visible helps officials diagnose issues effi-
ciently, minimizing downtime and ensuring a smooth competition experience. If the V5 Brain is
accessible, Field volunteers can help Teams troubleshoot time-sensitive issues prior to a
Match, including switching between Bluetooth and VEXnet radio modes as needed, selecting
programs on the V5 Brain in instances that prevent selection via the V5 Controller, etc. Teams
will also have easier access during any needed <GG4a> interactions.
<R8> Firmware. Teams must use VEXos version 1.1.5 or newer, found at https://link.vex.com/firmware. Custom
firmware modifications are not permitted.
a. The minimum version requirement is subject to change over the course of the season.
b. When the minimum version is updated, Teams have a one week (7 calendar day) grace period from the time
the minimum version is changed to update their firmware to the latest minimum version.
c. VEX reserves the right to deem any firmware update critical, and remove the allowable grace period.
d. Beta firmware, which includes any firmware version that ends with the letter ‘b’, is not legal for use in
competition.
<R9> Use a “Competition Template” for programming. The Robot must be programmed to follow control
directions provided by the VEXnet Field Controllers or Smart Field Control system.
During the Autonomous Period, Drive Team Members will not be allowed to use their V5 Controllers. As such,
Teams are responsible for programming their Robot with custom software if they want to perform in the Auton-
omous Period.
This may be tested in inspection, where Robots may be required to pass a functional “enable/disable” test. For
more information on this, Teams should consult the help guides produced by the developers of their chosen
programming software.

---

49
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<R10> Motors are limited. Robots may use any combination of V5 Smart Motors, 11W (276-4840) and/or 5.5W
(276-4842), within the following criteria:
a. The combined power of all motors (11W & 5.5W) must not exceed 88W. This limit applies to all motors on the
Robot, even those which are not plugged in.
b. V5 Smart Motors connected to Smart Ports are the only motors that may be used with a V5 Robot Brain.
The 3-wire ports may not be used to control motors of any kind.
Example A B C D E
Qty of 11W motors 8 7 6 5 0
Qty of 5.5W motors 0 2 4 6 16
This rule is applied differently for VEX U. See Rule <VUR11>.
<R11> Motors are limited for Subsystem 2 (see <R1>). Robots may use any combination of V5 Smart Motors,
11W (276-4840) and/or 5.5W (276-4842), in Subsystem 2 within the following criteria:
a. The combined power of all motors in Subsystem 2 must not exceed 55W.
b. Motors used in Subsystem 2 cannot provide power to any mechanism that is not a part of Subsystem 1.
i. Motors in Subsystem 2 cannot be toggled, engaged, or configured such that they are capable of provid-
ing power to any part of the Robot that is not a part of Subsystem 1.
c. Teams may be required to demonstrate what each motor is capable of powering on their Robot during
inspection in order to satisfy this requirement.
This rule does not apply to VEX U. See <VUR11>.
<R12> Electrical power comes from VEX batteries only. Robots may use one (1) V5 Robot Battery (276-4811)
to power the V5 Robot Brain.
a. No other sources of electrical power are permitted, unless used as part of a non-functional decoration per
<R23e>.
b. There are no legal power expanders for the V5 Robot Battery.
c. V5 Robot Batteries may only be charged by a V5 Robot Battery Charger (276-4812 or 276-4841).
d. V5 Controllers (276-4820) may only be powered by their internal rechargeable battery.
i. Teams are permitted to have an external power source (such as a rechargeable battery pack) plugged
into their V5 Controller during a Match, provided that this power source is connected safely and does
not violate any other rules, such as <R28>.
ii. Some events may choose to provide Field power for V5 Controllers. If this is provided for all Teams at
the event, then this is a legal power source for the V5 Controllers.
This rule is applied differently for VEX U. See Rule <VUR12>.

---

50
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<R13> Robots use VEXnet. Robots must ONLY utilize VEXnet for all wireless Robot communication.
a. Electronics from the Cortex, VEX EXP, VEX CTE. VEXpro, VEX RCR, VEXplorer, VEX IQ, VEX GO, or VEX
Robotics by HEXBUG product line are prohibited unless otherwise noted in <R16>.
b. Teams are permitted to use the Bluetooth® capabilities of the V5 Robot Brain and/or V5 Controller in Team
pits, practice Fields, and Robot Skills Matches. However, VEXnet must be used for wireless communication
during Head-to-Head Matches.
c. Teams are permitted to use the Wi-Fi capabilities of the Vision Sensor in Team pits or outside of Matches.
However, the Vision Sensor must have its wireless transmitting functionality disabled during Matches.
This rule is applied differently for VEX U. See Rules <VUR2>, <VUR10> and <VUR12>.
<R14> Give the radio some space. The V5 Radio should be mounted such that no metal surrounds the radio
symbol on the V5 Radio. While failure to do so will affect Robot performance, it will not prevent the Robot from
passing inspection.
It is fine to loosely encapsulate the V5 Radio within Robot structure. The intent of this rule is
to minimize radio connection issues by minimizing obstructions between VEXnet devices.
Burying a radio deep within a Robot may result in Robot communication issues. It is also recom-
mended that the LEDs on the radio be visible to aid in troubleshooting.
<R15> One or two Controllers per Robot. No more than two (2) VEX V5 Controllers may control a single Robot.
a. No physical or electrical modification of these Controllers is allowed under any circumstances.
i. Attachments which assist the Drive Team Member in holding or manipulating buttons/joysticks on the
V5 Controller are permitted, provided that they do not involve direct physical or electrical modification
of the Controller itself.
b. No other methods of controlling the Robot (light, sound, etc.) are permissible.
i. Using sensor feedback to augment driver control (such as motor encoders or the Vision Sensor) is
permitted.
<R16> Robots are built from the VEX V5 system. Robots may be built ONLY using official VEX
V5 components, unless otherwise specifically noted within these rules.
a. All legal parts for the V5 system are listed in the V5 Competition Legal Parts List.
b. Any questions or concerns about legal parts should be directed to the official Q&A System on events.vex.
com.
This rule is applied differently for VEX U. See Rule <VUR2>.
<R17> New VEX parts are legal. Additional VEX components released during the competition season on www.
vexrobotics.com are considered legal for use unless otherwise noted.
Some “new” components may have certain restrictions placed on them upon their release. These restrictions
will be documented in the official Q&A, in a Game Manual update, or in the VEX V5 Legal Parts List.

---

51
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<R18> Prohibited Items. The following types of mechanisms and components are NOT allowed.
a. Those that could potentially damage Field Elements or Scoring Objects.
b. Those that could potentially damage other competing Robots.
c. Those that pose an unnecessary risk of Entanglement with other Robots or Field Elements.
d. Those that could pose a potential safety hazard to Drive Team Members, event staff, or other humans.
e. Products from the VEXpro, VEX EXP, VEX IQ, VEX GO, VEX 123, VEX CTE, VEX AIM, VEX AIR, VEX Robotics
by HEXBUG*, or any other VEX product lines, unless specifically allowed by a clause of <R16> or listed in the
VEX V5 Legal Parts List (See <R16>).
f. Components obtained from the V5 beta program. All V5 beta hardware can be identified by its lighter gray
pre-production color. Robot Brains, Robot Batteries, Controllers, and Vision Sensors from the V5 beta have
a “BETA TEST” stamp on them. Smart Motors and Radios do not have this stamp, but can still be identified
by color.
g. Standalone VEX Smart Field Controller Brains (SKU 276-7577).
h. VEX apparel, competition support materials, packaging, or other non-Robot products.
i. Field Elements or components of Field Elements that are not otherwise explicitly legal materials (e.g.,
screws, standoffs, nuts, or non-shattering plastic within the limitations of <R24>).
j. 3D printed Robot parts for any purpose, including non-functional decorations and license plates.
k. Speakers and other audio devices that create sound.
3D printed Controller attachments, 3D printed Robot alignment tools, and/or other custom 3D
printed tools that do not go onto the Robot or into the Field during a Match are not considered
Robot parts, and may be legal for use if they meet the requirements of other pertinent rules.
This rule is applied differently for VEX U. See Rules <VUR2>, <VUR5>, <VUR6>, & <VUR13>.
* The HEXBUG brand is a registered trademark belonging to Spin Master Corp
<R19> Certain non-VEX components are allowed. Robots are allowed the following additional components
from sources other than VEX Robotics:
a. Any non-aerosol-based grease or lubricating compound, when used in extreme moderation on surfaces
and locations that do NOT contact the playing Field walls, foam Field tiles, Scoring Objects, or other Robots.
Grease or lubricant applied directly to V5 Smart Motors or Smart Motor cartridges is prohibited.
b. Anti-static compound, when used in extreme moderation (i.e., such that it does not leave residue on Field
Elements, Scoring Objects, or other Robots).
c. Hot glue when used to secure cable connections.
d. An unlimited amount of non-elastic rope/string, no thicker than 1/4” (6.35mm).
e. Commercially available items used solely for bundling or wrapping of 2-wire, 3-wire, 4-wire, or V5 Smart
Cables, and/or pneumatic tubing are allowed. These items must solely be used for the purposes of cable/
tubing protection, organization, or management. This includes but is not limited to electrical tape, cable
carrier, cable track, etc. It is up to inspectors to determine whether a component is serving a function
beyond protecting and managing cables and tubing.
f. Rubber bands no larger than 7.5” long and 0.25” wide.

---

52
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
g. Anti-slip drawer liner with similar nominal dimensions to VEX Anti-Slip Mat (275-0120 or 275-0121). The
pattern of the matting should be similar to VEX Anti-Slip Mat, and there should be no additional perfor-
mance gained by using non-VEX anti-slip drawer liner (i.e., no additional elasticity, adhesive backing, etc.).
No piece of anti-slip drawer liner can be larger than 12” x 15” and all anti-slip drawer liner must be black in
color.
h. Plastic zip ties no larger than 12” long and 0.25” wide.
i. A Micro SD card installed in the V5 Robot Brain.
j. Aerosol-based cooling/freeze spray may be used to assist in cooling motors. Teams using freeze spray or
similar products in ways that may reasonably be deemed unsafe could be subject to <S1> Violations.
k. Cleaners, disinfectants, and/or sanitizers may be used to assist in cleaning Robots, parts, components, etc.
l. See rules <R21> through <R25> for additional legal non-VEX components.
This rule is applied differently for VEX U. See Rules <VUR3>, <VUR4>, <VUR7>, <VUR8>,
<VUR9>, <VUR12>, <VUR14> & <VUR15>.
<R20> Custom V5 Smart Cables are allowed. Teams who create custom cables acknowledge that incorrect
wiring may have undesired results.
a. Official V5 Smart Cable Stock must be used.
b. Use of non-VEX 4P4C connectors and 4P4C crimping tools is permissible.
c. V5 Smart Cables may only be used for connecting legal electronic devices to the V5 Robot Brain.
<R21> A limited amount of tape is allowed. Robots may use a small amount of tape for the following
purposes:
a. To secure any connection between the ends of two (2) VEX cables.
b. To label wires, motors, Robot Brains, and/or controllers.
c. To prevent leaks on the threaded portions of pneumatic fittings. This is the only acceptable use of Teflon
tape.
d. In any other application that would be considered a “decoration” per <R23>.
e. As an aglet at the end of rope/string to prevent fraying.
<R22> Certain non-VEX fasteners are allowed. Robots may use the following commercially available
hardware:
a. #4, #6, #8, M3, M3.5, or M4 screws up to 2.5” (63.5 mm) long, and M2.5 x 8mm screws.
b. Shoulder screws with a shoulder length no longer than 0.20” and a diameter no larger than 0.176”.
c. Any commercially available nut, washer, threaded standoff up to 6” (152.4mm) long, and/or non-threaded
spacer up to 2.5” (63.5mm) long which fits these screws.
The intent of the rule is to allow Teams to purchase their own commodity hardware without
introducing additional functionality not found in standard VEX equipment. It is up to inspectors
to determine whether the non-VEX hardware has introduced additional functionality or not.

---

53
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
For the purposes of this rule, weight savings is not considered additional functionality.
If a key component of a Robot’s design relies upon convincing an inspector that a specialized
component is “technically a screw,” it is probably outside of the spirit and intent of this rule.
All specific dimensions listed in this rule are intended to be ‘nominal’ references to hardware
sizes found within the VEX V5 product line and/or their metric equivalents.
This rule is applied differently for VEX U. See Rule <VUR9>.
<R23> Visual decorations are allowed. Teams may add non-functional decorations, provided that they do
not affect Robot performance in any significant way or affect the outcome of the Match. These decorations
must be in the spirit of the competition. Inspectors and Head Referees will have final say in what is considered
“non-functional.” Unless otherwise specified below, non-functional decorations are governed by all standard
Robot rules.
To be considered “non-functional,” decorations must be backed by legal materials that provide the same
functionality. For example, if a Robot has a giant decal that prevents Scoring Objects from falling out of the
Robot, the decal must be backed by VEX material that would also prevent the Scoring Objects from falling out.
A simple way to check this is to determine if removing the decoration would impact the performance of the
Robot in any way.
a. Anodizing, painting, dyeing, or otherwise changing the color of any legal VEX part is considered a legal
nonfunctional decoration.
Note: For official events held in mainland China, anodizing, painting, dyeing, or otherwise changing the
color of any legal VEX part is prohibited.
b. Small cameras are permitted as non-functional decorations, provided that any transmitting functions or
wireless communications are disabled. Unusually large cameras used as ballast are not permitted.
c. VEX electronics may not be used as non-functional decorations.
d. Decorations that visually mimic Field Elements or Scoring Objects are not permitted. The inspector and
Head Referee will make the final decision on whether a given decoration or mechanism violates this rule.
i. Decorations cannot be designed to intentionally interfere with opponent Vision or Color Sensors.
Teams will be asked to remove these decorations if the inspector or Head Referee determines that it
interferes with opponent systems.
e. Decorations which provide feedback to a Robot (e.g., by influencing legal sensors) would be considered
“functional,” and are not permitted.
f. Decorations that cover or obscure identifying features of electronics and/or pneumatics parts are not legal.
i. Teams will be asked to either replace the electronics and/or pneumatics part entirely, or remove the
decoration if possible.
ii. Identifying features include, but are not limited to, VEX logos, part numbers, and other distinctive colors
or features of the part that allow an inspector to easily confirm it is a legal part.
<R24> A limited amount of custom plastic is allowed. Robots may use custom-made pieces cut from certain
types of non-shattering plastic, up to 0.070” (1.78mm) thick.

---

54
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
a. Each Robot is limited to a maximum of 12 individual pieces cut from non-shattering plastic. This includes
non-shattering plastic used in non-functional decorations.
b. Each individual piece of non-shattering plastic cannot be larger than 4” x 8” x 0.070”.
c. Teams must present and display ALL non-shattering plastic parts during inspection.
i. Inspectors will verify the total number of plastic pieces. They may use dry-erase markers or other forms
of temporary marking to aid in counting.
ii. Inspectors will verify that no non-shattering piece exceeds the size limitation.
d. Plastic may be mechanically altered by cutting, drilling, bending, etc. It cannot be chemically treated,
melted, cast, or bonded to another part. Heating non-shattering plastic to aid in bending is acceptable.
e. Legal plastic types are polycarbonate (Lexan), acetal monopolymer (Delrin), acetal copolymer (Acetron GP),
POM (acetal), ABS, PEEK, PET, HDPE, LDPE, nylon (all grades), polypropylene, PTFE, and FEP.
f. Shattering plastics, such as PMMA (also called plexiglass, acrylic, or perspex), and structurally non-homo-
geneous plastics, such as woven (e.g., SRPP) and/or fiber-reinforced sheets, are prohibited.
g. Plastic sheets sold by VEX (such as the 276-8340 PET Sheets) are considered custom plastic in the context
of this rule, and are subject to the same limitations as other plastic parts.
h. This rule does not apply to 3D printed plastic parts. 3D printed Robot parts are not permitted in the VEX V5
Robotics Competition for any purpose, including non-functional decorations.
Note: Teams are strongly encouraged to provide inspectors with 1:1 scale drawings, identical spares,
or 1:1 scale tracings of their non-shattering plastic pieces to aid in inspection. Drawings and tracings
should accurately reflect ALL shapes and dimensions of each piece. Complex bent/cut parts, and parts
that are cut diagonally across a 4”x8” sheet to gain length may require more evidence.
This rule is applied differently for VEX U. See Rules <VUR3> & <VUR4>.
Violation Notes:
1. Teams must replace broken plastic pieces that result in temporary, unintentional <R24>
Violations.
<R25> Pneumatics are limited. A Robot’s pneumatic subsystem must satisfy all of the following criteria:
a. Teams may use a maximum of two (2) VEX Air Tanks (276-8749) on a Robot.
b. Pneumatic devices may be charged to a maximum of 100 psi.
c. The compressed air contained inside a pneumatic subsystem can only be used to actuate legal pneumatic
devices (e.g., cylinders).
The intent of <R25a> and <R25b> is to limit Robots to the air pressure stored in two reservoir
tanks, as well as the normal working air pressure contained in their pneumatic cylinders and
tubing on the Robot. Teams may not use other elements for the purposes of storing or gener-
ating air pressure.
Using cylinders or additional pneumatic tubing solely for additional storage is a Violation of the
spirit of this rule. Similarly, using pneumatic cylinders and/or tubing without any air reservoirs is
also a Violation of the spirit of this rule.

---

55
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
The intent of <R25c> is to ensure that pneumatics are being used safely. Pressurized systems,
such as a Robot’s pneumatic subsystem, have the potential to be dangerous if used incorrectly.
This rule ensures the safety of participants, and prevents potentially unsafe uses in the future.
Another way of thinking of <R25c> is that “pneumatics should only be used with pneumatics.”
Teams should not use compressed air as a means of actuating non-pneumatic devices such as
screws, nuts, etc. For example, pulling a pin with a pneumatic cylinder is okay, but using air to
actuate the pin itself is not.
This rule is applied differently for VEX U. See Rule <VUR14>.
<R26> The VEX Pressure Gauge (276-8748) is a required part if a Robot includes pneumatics.
a. The gauge must be plumbed to the high pressure side of the system if the Team is using a regulator.
b. The pressure gauge must be visible and readable by referees and inspectors without moving or removing
other Robot components or mechanisms.
c. The pressure gauge will be used to verify that the pneumatics system is charged to no more than 100psi,
per <R25>.
i. The VEX Pressure Gauge on the Robot will take precedence over any off-board measurement.
<R27> Most modifications to non-electrical components are allowed. Physical modifications, such as
bending or cutting, of legal metal structure or plastic components are permitted.
a. Modifying the arm of limit switches (by bending or trimming) or removing the rubber cap of bumper switches
is permitted.
b. Metallurgical modifications that change fundamental material properties, such as heat treating or melting,
are not permitted.
c. Pneumatic tubing may be cut to desired lengths.
d. Fusing/melting the end of legal nylon rope/string (see <R19d>) to prevent fraying is permitted.
e. Welding, soldering, brazing, gluing, or attaching parts to each other in any way that is not provided within the
VEX platform is not permitted. Rule <R19c> is an exception to this rule.
f. Mechanical fasteners may be secured using Loctite or a similar thread-locking product. This may ONLY be
used for securing hardware, such as screws and nuts.
This rule is applied differently for VEX U. See Rules <VUR3>, <VUR12> & <VUR13>.
<R28> No modifications to electronic or pneumatic components are allowed. Motors (including the V5
Smart Motor firmware), microcontrollers (including V5 Robot Brain firmware), cables, sensors, controllers,
battery packs, reservoirs, solenoids, pneumatic cylinders, and any other electrical or pneumatics component
of the VEX platform may NOT be altered from their original state in ANY way.
a. Teams may make the following modifications to the V5 Smart Motor (11W)’s user-serviceable features. This
list is all-inclusive; no other modifications are permitted. Where applicable, the components listed below (in
the specific applications listed below) are permissible exceptions to <R19>.
i. Replacing the gear cartridge with other official cartridges.

---

VEX V5 Robotics Competition Override - Game Manual
56
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
ii. Removing or replacing the screws from the V5 Smart Motor Cap (276-6780).
iii. Removing or replacing the threaded mounting inserts (276-6781).
iv. Aesthetic/non-functional labeling (e.g., markers, stickers, paint, etc.).
b. V5 Smart Motors (11W) must use an official VEX V5 gear cartridge. For the purposes of this rule, the gear
cartridges found within the V5 Smart Motor are considered “part of the motor.” Therefore, any physical or
functional modifications to official gear cartridges is not permitted. V5 Smart Motors (11W) may only use
official VEX motor cartridges
c. For the purposes of this rule, the V5 Smart Motor Cap is not considered “part of the motor.” Therefore,
<R27> applies.
d. External wires on VEX 2-wire or 3-wire electrical components may be repaired by soldering or using twist/
crimp connectors, electrical tape, or shrink tubing such that the original functionality and length are not
modified in any way.
i. Wire used in repairs must be identical to VEX wire.
ii. Teams make these repairs at their own risk; incorrect wiring may have undesired results.
e. V5 Robot Brain accessories (short flanges, long flanges, and the magnetic screen protector) are consid-
ered “part of the V5 Robot Brain” and cannot be modified.
f. The modifications listed in clause A also apply to equivalent parts for the 276-4842 V5 Smart Motor (5.5W),
with the exception of gear cartridges which only apply to the V5 Smart Motor (11W). For the purposes of
this rule, the motor cap for the V5 Smart Motor (5.5W) is not considered “part of the motor,” and therefore
<R27> applies.

---

Copyright 2026, VEX Robotics, Inc.
## Section 5
## The Tournament

---

58
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Overview
# Section 5 - The Tournament
# Tournament Rules
The VEX V5 Robotics Competition consists of Head-to-Head Matches, Robot Skills Matches, and optional
judging. This section describes how Head-to-Head Matches and Robot Skills Matches are to be played at a
given event.
Awards may be given to top Teams in each format, as applicable. Awards may also be given for overall perfor-
mance in the judged criteria.
<T1> Head Referees have ultimate and final authority on all gameplay and Robot ruling decisions during
the competition.
a. Scorekeeper Referees score the Match, and may serve as observers or advisers for Head Referees, but
may not determine any Violations directly.
b. When issuing a Major Violation or Minor Violation to a Team, Head Referees must provide the rule number
of the specific rule that has been Violated, and must record the Violation on the Match Anomaly Log.
c. Event Partners may not overrule a Head Referee’s gameplay or Robot decisions.
d. Every Qualification Match and Elimination Match must be watched by a certified Head Referee. A Head
Referee may only watch one Match at a time; if multiple Matches are happening simultaneously on separate
Fields, each Field must have its own Head Referee. Head Referees must follow the rules in this game manual
and the Q&A, and must make rulings consistent with the intent of the game manual and Q&A.
e. At a minimum, every Robot Skills Match must be watched by a trained Scorekeeper Referee, who may only
watch one Match at a time. If multiple Robot Skills Matches are happening simultaneously on separate
Fields, each Field must have its own Scorekeeper Referee. A certified Head Referee must be available at
the event to explain a rule, Disqualification, Violation, or other penalty to Teams in Robot Skills Matches as
needed in support of the Scorekeeper Referees at skills Fields.
Note from the VEX GDC: The rules contained in this Game Manual are written to be enforced by
human Head Referees. Many rules have “black-and-white” criteria that can be easily checked.
However, some rulings will rely on a judgment call from this human Head Referee. In these
cases, Head Referees will make their calls based on what they and the Scorekeeper Referees
saw, what guidance is provided by their official support materials (the Game Manual and the
Q&A), and most crucially, the context of the Match in question.
The VEX V5 Robotics Competition does not have video replay, our Fields do not have absolute
sensors to count scores, and most events do not have the resources for an extensive review
conference between each Match.
When an ambiguous rule results in a controversial call, there is a natural instinct to wonder
what the “right” ruling “should have been,” or what the Game Design Committee “would have
ruled.” This is ultimately an irrelevant question; our answer is that when a rule specifies “Head
Referee’s discretion” (or similar), then the “right” call is the one made by a Head Referee in the
moment. The VEX GDC designs games, and writes rules, with this expectation (constraint) in
mind.

---

59
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T2> Head Referees must be qualified. V5RC Head Referees must have all of the following qualifications:
a. Be at least 20 years of age.
b. Be approved by the Event Partner.
c. Be a certified V5RC Head Referee for the current season.
d. Cannot be the Event Partner or a Judge Advisor for the event.
Note: Scorekeeper Referees must be at least 15 years of age, and must be approved by the Event
Partner.
Head Referees should demonstrate the following attributes:
● Thorough knowledge of the current game and rules of play
● Effective decision-making skills
● Attention to detail
● Ability to work effectively as a member of a team
● Ability to be confident and assertive when necessary
● Strong communication and diplomacy skills
<T3> Drive Team Members are permitted to immediately appeal a Head Referee’s ruling. If Drive Team
Members wish to dispute a score or ruling, they must stay in the Alliance Station until the Head Referee from
the Match talks with them. The Head Referee may then choose to meet with the Drive Team Members at
another location and/or at a later time so that the Head Referee has time to reference materials or resources to
help with the decision. Once the Head Referee announces that their decision has been made final, the issue is
over and no more appeals may be made (See rule <T1>); failure to accept this final decision may be considered
a <G1> Violation. There is no system or opportunity for an appeal of the Head Referee’s final decision, either at
or after the event.
a. Referees are not permitted to review any photo or video Match recordings when determining a score or
ruling. Some events may also prohibit Drive Team Members from reviewing photo or video Match record-
ings while in the Alliance Station; this should be announced to all Teams before Matches start.
b. Head Referees are the only individuals permitted to explain a rule, Disqualification, Violation, or other
penalty to the Teams in a Head-to-Head Match. Teams should never consult other field personnel, including
Scorekeeper Referees, regarding a ruling clarification.
Communication and conflict resolution skills are an important life skill for Students to practice and learn. In the
VEX V5 Robotics Competition, we expect Students to practice proper conflict resolution using the proper chain
of command. Violations of this rule may be considered a Violation of <G1>.
Some events may choose to utilize a “question box” or other designated location for discus-
sions with Head Referees. Offering a “question box” is within the discretion of the Event Partner
and/or Head Referee, and may act as an alternate option for asking Drive Team Members to
remain in the Alliance Station (although all other aspects of this rule apply).
However, by using this alternate location, Drive Team Members acknowledge that they are
forfeiting the opportunity to use any contextual information involving the specific state of the
Field at the end of the Match. For example, it is impossible to appeal whether a Scoring Object
was Scored or not if the Field has already been reset. If this information is pertinent to the

---

60
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
appeal, Drive Team Members should still remain in the Alliance Station, and only relocate to
the “question box” once the Head Referee has been made aware of the concern and/or any
relevant context.
<T4> The Event Partner has ultimate authority regarding all non-gameplay decisions during an event. The
Game Manual is intended to provide a set of rules for successfully playing V5RC Override; it is not intended to
be an exhaustive compilation of guidelines for running a VEX V5 Robotics Competition event. Rules such as,
but not limited to, the following examples are at the discretion of the Event Partner and should be treated with
the same respect as the Game Manual.
● Venue access
● Pit spaces
● Health and safety
● Team registration and/or competition eligibility
● Team conduct away from competition Fields
This rule exists alongside <G1>, <S1>, and <G3>. Even though there isn’t a rule that says “do not
steal from the concession stand,” it would still be within an Event Partner’s authority to remove
a thief from the competition.
<T5> Be prepared for minor Field variance. Field Element tolerances and Scoring Objects may vary from
specified locations/dimensions; Teams are encouraged to design their Robots accordingly. Please make sure
to check Appendix A for more specific nominal dimensions and tolerances.
a. Field Element tolerances may vary from nominal by up to ±1.0”.
b. Scoring Object placement at the beginning of the Match may vary from nominal by up to ±1” (25.4mm). If a
Scoring Object is within tolerance, it should not be adjusted before the Match.
c. The rotation of Scoring Objects is not specified. If a Scoring Object is within tolerance, either on the Field or
within a Loader, it should not be adjusted before the Match.
The Field Perimeter and Field Elements are designed to be assembled and disassembled multiple times each
year. Event Partners store and transport Fields between events, and the individuals setting up the Field at one
event may differ from those at the next. While every effort will be made to ensure minimal variance, Teams
should expect that any Field may be slightly different than another, and prepare accordingly. Just because
something works on one Field does not fully guarantee it will work on the next, and is not enough evidence
alone to determine if a Field is out of tolerance.
<T6> Fields may be repaired at the Event Partner’s discretion. All competition Fields at an event must be
set up in accordance with the specifications in Appendix A and/or other applicable Sections. Minor aesthetic
customizations or repairs are permitted, provided that they do not impact gameplay (see <T4>).
Examples of permissible modifications include, but are not limited to:
● Applying threadlocker to Field Element mounting hardware
● Using non-VEX electrical tape to add required lines to the Field
● Anchoring Field Elements directly to Field risers instead of the metal plates

---

61
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
● Anchoring the metal plates to the underlying surface with hardware or tape
● Utilizing a non-VEX Strap under the Field to tie walls together
● Parts added to the exterior of the Portable Field Perimeter to assist in assembly, transportation, and rigidity
during Matches
● Custom solutions for mounting Toggle brackets to metal Field Perimeters. See Appendix A drawing titled
“Toggle Mounting - Metal Field Perimeter” for examples.
● Adjust the size/position of the Alliance Station tape lines.
Note: Exact dimensions/positioning of the Alliance Stations are subject to Event Partner discretion.
Alliance Station tape lines should always be placed such that Drive Team Members can reasonably view
and interact with the Field from within the Alliance Station during a Match (e.g., Drive Team Members can
easily reach Loaders from within the Alliance Station).
Modifications that may impact Robot functionality and/or how the game is played are generally not allowed.
Examples of prohibited modifications include, but are not limited to:
● Unofficial Field Perimeter walls, additional structural elements inside of the Field Perimeter, or unofficial/
replica Field Elements
● Additional VEX structural parts attached to a Field Element
● Replacing the opaque Field walls on the VEX Portable Competition Field Perimeter with transparent panels
● Assembling a VEX Portable Competition Field Perimeter without including the securing straps
● Affixing stickers to the foam Field Tiles or otherwise marking object placements for Field reset
Any specific repairs and/or modifications which pertain to the current season’s game will be documented in
this rule and Appendix A, as needed.
<T7> Fields at an event must be consistent with each other. There are many types of permissible aesthetic
and/or logistical modifications that may be made to competition Fields at the Event Partner’s discretion (see
<T6>). If an event has multiple Head-to-Head competition Fields, they must all incorporate the same permissi-
ble/applicable modifications. If an event has multiple Robot Skills Challenge Fields, they must all incorporate
the same permissible modifications. For example, if one Head-to-Head competition Field is elevated, then all
Head-to-Head competition Fields must be elevated to the same height.
Examples of these modifications may include, but are not limited to:
● Elevating the playing Field off of the floor (common heights are 12” to 24” [30.5cm to 61cm])
● Field control systems (see <T8>)
● Field display monitors
● Field Perimeter decorations (e.g., LED lights, sponsor decals on polycarbonate panels)
● Field Perimeter type (see <T9>)
● Utilizing the VEX GPS Field Code Strips
● Utilizing non-VEX Straps to hold the Field walls together
Note: If an event has dedicated Fields for Robot Skills Matches, there is no requirement for them to have
the same consistent modifications as the Head-to-Head Fields. See <T21> for more details.

---

62
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T8> There are three types of Field control that may be used.
1. A VEXnet Field Controller controlled by Tournament Manager, which connects to a Controller’s competition
port via ethernet cable.
2. A V5 Event Brain controlled by Tournament Manager, which connects to a Controller via Smart Cable.
3. A VEXnet Competition Switch, which connects to a Controller’s competition port via Cat-5 cable, may only
be used in Practice Matches, Robot Skills Matches, and Leagues, and only under extreme circumstances.
If an event has multiple Fields, then all Fields of the same game type must use the same control system, in
accordance with <T7> and <T21>. For example, it would be permissible for Head-to-Head competition Fields to
use V5 Event Brains, and for Skills Challenge Fields to use VEXnet Field Controllers. However, it would not be
permissible for one Head-to-Head Field to use a V5 Event Brain while another Head-to-Head Field uses a VEXnet
Field Controller.
Note: Official Qualifying Events may only use the official, unmodified version of Tournament Manager
for field control, along with approved hardware and networking solutions. Add-ons that abide by the TM
Public API guidelines are permitted. Once add-ons are enabled, the software is no longer supported by
VEX Robotics, or DWAB Technologies; any necessary troubleshooting will be done at the user’s own risk.
<T9> There are two types of Field Perimeter that may be used.
1. VEX Metal Competition Field Perimeter (SKU 278-1501).
2. VEX Portable Competition Field Perimeter (SKU 276-8242).
See Appendix A for more details.
If an event has multiple Fields, then all Fields of the same game type must use the same Field Perimeter type, in
accordance with <T7> and <T21>. For example, it would be permissible for Head-to-Head competition Fields to
use metal Field Perimeters, and for Skills Challenge Fields to use Portable Field Perimeters.
However, it would not be permissible for one Head-to-Head Field to use a metal Field Perimeter, while other
Head-to-Head Fields use Portable Field Perimeters.
<T10> Qualification Matches follow the Match Schedule. A Qualification Match Schedule will be available on
the day of competition. The Match Schedule will indicate Alliance partners, Match pairings, and Alliance colors
for each Match. For Tournaments with multiple Fields, the schedule will indicate which Field each Match will take
place on.
a. Practice Matches may be included in the Match Schedule at some events, but are not required. If Practice
Matches are run, every effort will be made to equalize practice time for all Teams.
b. A Qualification Match can only start before its scheduled time if all Teams, Robots, and assigned volunteers
are at the Field and ready to play.
c. Any multi-division event must be approved by VEX Robotics prior to the event, and divisions must be
assigned in sequential order by Team number.
d. The Match Schedule is subject to change at the Event Partner’s discretion. Events should generally wait to
generate the Match Schedule until all Teams have checked in and passed Robot Inspection, or when it has
been confirmed that Teams will not be participating.

---

63
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
e. If it is determined by both the Event Partner and VEX Robotics that a Team should be removed from further
Matches at the event based on the outcome of a serious Violation, the Event Partner can use the full-event
Disqualification feature in TM to remove that Team from all further Matches at the event. Their spots in later
Qualification Matches will be replaced automatically by Teams from the end of the Match Schedule, and
Matches will be removed from the end of the schedule as appropriate. The Event Partner cannot do this
without permission from VEX Robotics.
<T11> Each Team will have at least six Qualification Matches.
a. A Tournament must include a minimum of six Qualification Matches per Team at local qualifying events or
eight Qualification Matches per Team at a Championship event. The recommended number of Qualification
Matches per Team is eight for a standard Tournament, and up to ten for a championship event.
b. A league must include at least three league ranking sessions, with at least one week between sessions.
Each session must include a minimum of two Qualification Matches per Team at that session. The suggest-
ed number of Qualification Matches per Team for a standard league ranking session is four. Leagues will
have a championship session where elimination rounds will be played. Event Partners may also choose to
have Qualification Matches as part of their championship session.
<T12> Qualification Matches contribute to a Team’s ranking for Alliance Selection.
a. When in a Tournament, every Team will be ranked based on the same number of Qualification Matches.
b. For Tournaments that have more than one division, Teams will be ranked among all Teams in their specific
division.
c. When in a league, every Team will be ranked based on the number of Matches played. Teams that partici-
pate in at least 60% of the total Matches available will be ranked above Teams that participate in less than
60% of the total Matches available; e.g., if the league offers 3 ranking sessions with 4 Qualification Matches
per Team, Teams that participate in 8 or more Matches will be ranked higher than Teams who participate in 7
or fewer Matches. Being a no-show to a Match that a Team is scheduled in still constitutes participation for
these calculations.
d. In some cases, a Team will be asked to play an additional Qualification Match. The extra Match will be iden-
tified on the Match Schedule with an asterisk; Win Points, Autonomous Points, and Strength of Schedule
Points for that Qualification Match will not impact a Team’s ranking, and will not affect participation percent-
age for leagues.
i. Teams are reminded that <G1> is always in effect and Teams are expected to behave as if the additional
Qualification Match counted.
ii. In leagues, Teams may have a different number of Qualification Matches. Rankings are determined by
the Win Percentage, which is the number of wins divided by the number of Qualification Matches that
Teams have played.
<T13> Qualification Match tiebreakers. Team rankings are determined throughout Qualification
Matches as follows:
1. Average Win Points (Win Points / number of Matches played)
2. Average Autonomous Points (Autonomous Points / number of Matches played)

---

64
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
3. Average Strength of Schedule Points (Strength of Schedule Points / number of Matches played)
4. Highest Match score
5. Second-highest Match score
6. Random electronic draw
<T14> Small Tournaments have fewer Alliances. The number of Alliances for a given event is determined as
follows, except in extraordinary circumstances with the permission of VEX Robotics:
# of Teams # of Elimination Alliances
32+ 16
24-31 12
16-23 8
<16 # of Teams divided by 2, less any remainder
This rule is applied differently for VEX U. See rule <VUT6>.
<T15> Send a Student representative to Alliance Selection. Each Team must send one Student representa-
tive to the playing Field (or other designated area) to participate in Alliance Selection. If the Team representative
fails to report in for Alliance Selection, their Team will be ineligible for participation in the Alliance Selection
process.
Once the Alliance Selection begins, Student representatives cannot use electronic devices unless they have
been demonstrated to be in airplane mode. No electronic communication by or with Student representatives is
allowed during the Alliance Selection process.
Teams are advised to complete their scouting prior to the beginning of Alliance Selection, and to come to
Alliance Selection prepared with a written list of potential Alliance Partners. Non-electronic methods of com-
munication are allowed. Rule <G2> and the Student-Centered Policy still apply during Alliance Selection. Any
communication about Alliance Selection and specific Teams should be limited to Student Team Members.
<T16> Each Team may only be invited once to join one Alliance. If a Team representative declines an Alliance
Captain’s invitation during Alliance Selection, that Team is no longer eligible to be selected by another Alliance
Captain. However, they are still eligible to play Elimination Matches as an Alliance Captain. This video includes a
full explanation of the Alliance Selection process.
For example:
● Alliance Captain 1 invites Team 000A to join their Alliance.
● Team 000A declines the invitation.
● No other Alliance Captains may invite Team 000A to join their Alliance.
● However, Team 000A may still form their own Alliance if Team 000A is ranked high enough after Qualifica-
tion Matches to become an Alliance Captain.
Note: Alliances must have two Teams, and there are no “do-overs” during Alliance Selection. If enough
Teams decline their invitations such that the full number of Alliances cannot be filled, the event will
proceed with a reduced number of Alliances.

---

65
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T17> Elimination Matches follow the Elimination Bracket. A sixteen (16) Alliance bracket plays as shown in
Figure T17-1, with Matches proceeding in numbered order through each round.
If an event is run with fewer than 16 Alliances, then they will use the bracket shown in Figure T17-1, with Byes
awarded when there is no applicable Alliance. For example, in a Tournament with 12 Alliances, Alliances 1, 2, 3,
& 4 would automatically advance to the Quarterfinals.
Figure T17-1: A 16-Alliance bracket
Thus, an eight (8) Alliance bracket would run as shown below:
Figure T17-2: An 8-Alliance bracket
At events with multiple divisions, each division champion Alliance will advance to the overall event finals.
Alliance color assignment for these Matches will be determined by the original Alliance seeding within their
respective divisions. The higher seeded Alliance (e.g., 1 is higher than 2) will be designated as the red Alliance,
and the lower-seeded Alliance will be designated as the blue Alliance.
If both Alliances hold the same seed number within their divisions, the Event Partner will conduct a coin flip to
determine which Alliance is assigned to the red Alliance, with the remaining Alliance assigned to blue.

---

66
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T18> Elimination Matches are a blend of “Best of 1” and “Best of 3.” “Best of 1” means that the winning
Alliance in each Match advances to the next round of the Elimination Bracket. “Best of 3” means that the first
Alliance to reach two wins will advance.
See the flowchart below for more information.
Figure T18-1: The process for determining how Elimination Matches should be played.

---

67
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T19> Ties in Elimination Matches lead to limited rematches. In the case of tied Matches during Elimination
Matches, Tournament Manager will apply the following logic to determine which Alliance will progress to the
next round.
a. In a “Best of 1” Elimination Round, the higher-seeded Alliance will advance and be declared the winner
under the following guidelines.
i. After two (2) ties in a non-finals Match.
ii. After three (3) ties in a finals Match.
b. For single-division events or within a division: in a ”Best of 3” Elimination Round, the higher-seeded Alliance
will advance and be declared the winner under the following guidelines.
i. After three (3) ties in a round in which neither Alliance has yet won a Match (0-0).
ii. After two (2) ties in a round in which each Alliance has won a single Match (1-1).
c. For single-division events or within a division: after two (2) ties in a “Best of 3” Elimination Round in which
one Alliance has won a single Match (1-0), the Alliance with one (1) win will be declared the winner.
d. For a “Best of 3” overall Finals round at a multi-division event, Teams should continue to play tiebreaker
Matches until one Alliance has won two (2) Matches.
<T20> Skills Match Schedule. Teams play Robot Skills Matches on a first-come, first-served basis. Each Team
will get the opportunity to play exactly three (3) Driving Skills Matches and three (3) Autonomous Coding Skills
Matches at each Tournament and/or League Session.
Teams should review the event agenda and their Match Schedule to determine when the best possible time is
to complete their Robot Skills Matches. If the Robot Skills Challenge area closes before a Team has completed
all six (6) Robot Skills Matches, but it is determined that there was adequate time given, then the Team will
automatically forfeit those unused Matches.
<T21> There is no requirement that Skills Challenge Fields have the same consistent modifications as the
Head-to-Head Fields. For example, there is no requirement that all Skills Challenge Fields are elevated to the
same height as Head-to-Head Fields. However, all Skills Challenge Fields at a single event must use the same
type of Field control and Field Perimeter, as described in rules <T8> and <T9>.
It is strongly recommended/preferred that all Skills Challenge Fields are consistent with each other, but this
may not be the case in extreme circumstances. In order to use non-conforming Head-to-Head Fields for Skills
Challenge runs (e.g. during lunch), the following steps should be taken:
● Teams must be informed that the Head-to-Head Fields may have some differences from the Skills Chal-
lenge Fields (e.g., they might not have GPS strips).
● Teams must be given an opportunity to select which type of Field they want to use, i.e. they cannot be
required to use a Head-to-Head Field for any Skills Challenge run.
<T22> Skills Rankings at events. Teams will be ranked at an event based on the following scores and
tiebreakers:
1. Sum of highest Autonomous Coding Skills Match score and highest Driving Skills Match score.
2. Highest Autonomous Coding Skills Match score.

---

68
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
3. Second-highest Autonomous Coding Skills Match score.
4. Second-highest Driving Skills Match score.
5. Highest sum of Skills Stop Times (see rule <RSC5>) from a Team’s highest Autonomous Coding Skills Match
and highest Driving Skills Match (i.e., the Matches in point 1).
6. Highest Skills Stop Time from a Team’s highest Autonomous Coding Skills Match (i.e., the Match in point 2).
7. Third-highest Autonomous Coding Skills Match score.
8. Third-highest Driving Skills Match score.
9. If a tie cannot be broken after all above criteria, then the following ordered criteria will be used to determine
which Team had the “best” Autonomous Coding Skills Match:
i. Quantity of Scored Pins
ii. Number of Toggles set to a color other than yellow
10. If the tie still isn’t broken, the same process in Step 9 will be applied to each Team’s best Driving Skills
Match.
11. If the tie still isn’t broken, events may choose to allow Teams to have one more deciding Driving Skills
Match, to be ranked according to the standard criteria above, or declare both Teams the Robot Skills Chal-
lenge Winner.
<T23> Skills Rankings Globally. Teams will be ranked globally based on their Robot Skills scores from Tourna-
ments and Leagues that upload results to events.vex.com, according to the following tiebreakers:
1. Highest Robot Skills score (combined Autonomous Coding Skills Match and Driving Skills Match Score from
a single event).
2. Highest Autonomous Coding Skills Match score (from any event).
3. Highest sum of Skills Stop Times (see rule <RSC5>) from the Robot Skills Matches used for point 1.
4. Highest Skills Stop Time from the Autonomous Coding Skills Match used for point 2.
5. Highest Driving Skills Match score (from any event).
6. Highest Skills Stop Time from the Driving Skills Match score used for point 5.
7. Earliest posting of the highest Autonomous Coding Skills Match score. The first Team to post a score ranks
ahead of other Teams that post the same score at a later time, all else being equal.
8. Earliest posting of the highest Driving Skills Match score. The first Team to post a score ranks ahead of
other Teams that post the same score at a later time, all else being equal.

---

VEX V5 Robotics Competition Override - Game Manual
69
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<T24> Robot Skills at League Events. At league events in which Teams may submit Robot Skills Challenge
scores across multiple sessions, the Robot Skills scores (combined highest Autonomous Coding Skills Match
and Driving Skills Match scores) used for rankings will be calculated from Matches within the same session.
For example, consider the following scores for a hypothetical Team across two league event sessions:
Autonomous Coding
Skills Match 
Driving Skills Match Robot Skills Score
Session 1 50 60 110
Session 2 40 100 140
This Team would have a Robot Skills score of 140 for this event, and their scores from Session 2 would be used
for the Event and Global tiebreakers listed in <T22> and <T23>.

---

Copyright 2026, VEX Robotics, Inc.
## Section 6
## VEX U

---

71
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
## Introduction
# Section 6 - VEX U
While many colleges and universities already use the VEX V5 system in their academic classes, many more
have extensive manufacturing capabilities beyond the standard “VEX metal” library. Fabrication techniques like
machining and 3D printing are more common than ever in collegiate engineering programs, and we can’t wait
to see what VEX U Robotics Competition Teams from around the world are able to create under these more
advanced rules.
As in past years, the season will include a culminating VEX U event at the VEX Robotics World Championship,
along with regional Tournaments across the world. Participating schools will get the chance to prove their
abilities in front of thousands of future engineers and show off what truly makes their school remarkable. The
VEX U Robotics Competition is the perfect project-based supplement to many university level engineering
programs, and gives Students the unique opportunity to demonstrate their real-world skills to potential
employers, such as our event sponsors.
# Game, Robot, and Tournament Rules
The VEX U Robotics Competition uses the VEX V5 Robotics Competition Override Field with some minor
modifications. Anyone that has a V5RC Override Field can use it for a VEX U event or Team. Please consult
earlier sections of this game manual for the basic set of competition rules and details. All of the standard
rules apply, except for the modifications listed in this section. In the event of a conflict between rules, the
rules listed in this Section of the game manual and any rulings on the official VEX U Q&A take precedence.
# VURC Definitions
Additional Electronics - Any Sensor, processor, or other electronic component used in Robot construction,
and connected to the V5 Robot Brain, that is not sold by VEX Robotics. Examples include commercially-
available devices (e.g., Raspberry Pi) or custom devices designed and fabricated by the Team. See <VUR12>
and <VUR13> for more details.
Alliance - A grouping of two (2) Robots from the same Team that have been chosen by the Students to play
together during a given Match.
Electromechanical Assembly - A complex system composed of multiple off-the-shelf components, which
may include Sensors, mechanical parts, and actuators.
External Processor - A computing device or microcontroller that independently processes Sensor data
before sending it to the VEX V5 Brain.
Fabricated Part - Any component used in Robot construction that is fabricated by Team members. See
<VUR3> through <VUR7> for more details.

---

72
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Rule Modifications: Field Setup
Raw Stock - Stock materials purchased from third-party vendors that may be used to create Fabricated Parts.
See <VUR4> and <VUR5>.
Sensor - A device that detects and responds to changes in the environment, providing data to the Robot’s
control system.
The VURC playing Field is set up differently than a Head-to-Head VEX V5 Robotics Competition Override
Match, with the following modifications:
● The VEX GPS code strip must be installed on the Field.
● Modified Field layout. Only the Midfield Goal should begin the Match with a yellow/yellow Pin Placed in it.
● Modified Match Loads. Each Alliance gains an additional two yellow/yellow Pins that begin off the Field
as Match Loads.
० Red Alliance Match Loads: 10 Cups, 10 red/yellow Pins, and 3 yellow/yellow Pins
० Blue Alliance Match Loads: 10 Cups, 10 blue/yellow Pins, and 3 yellow/yellow Pins

---

73
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure VEXU-1: An overhead view of the Field setup for a VEX U Match.

---

74
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure VEXU-2: A side view of the Field setup for a VEX U Match.

---

75
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Rule Modifications: Game
<VUG1> Different Robot starting sizes. See <VUR1>.
<VUG2> Different Robot placement than rule <GG10>. The red Team has the right to place one Robot on
the Field first, followed by both blue Robots, and ending with the 2nd red Robot. This applies in Qualification
Matches and Elimination Matches. If this right is used, once a Team has placed a Robot on the Field, its position
cannot be readjusted prior to the Match.
<VUG3> Some electronic devices may be in motion or moving at the beginning of the Match. This includes
active cooling fans, spinning LIDAR modules, and/or other similar sensors or Additional Electronics. These
electronic devices should not initiate any form of motion for the entire Robot or any of its subsystems, and may
not directly interact with Scoring Objects and/or other Robots.
<VUG4> Different availability of Loaders. Drive Team Members may introduce Match Loads through their
Alliance-colored Loaders during both the Autonomous Period and Driver Controlled Period of the Match.
<VUG5> Different scoring during the Autonomous Period. For the purpose of determining the Autonomous
Bonus winner, the following are included in Autonomous Period scoring calculations:
a. Ownership and Scoring of yellow Pins in the Midfield
i. A yellow Pin that is Placed in the Midfield is Owned by the Alliance that ends the Autonomous Period
with a greater number of Robots in the Midfield.
b. Scoring of Robots ending in the Midfield
i. A Robot counts as ending in the Midfield if the Robot is at least partially within the Midfield at the end of
the Autonomous Period.
<VUG6> Different Autonomous Win Point criteria. For all VEX U events, an Autonomous Win Point is awarded
to any Alliance that ends the Autonomous Period with all of the following tasks completed, and that has
committed no Violations during the Autonomous Period:
1. At least twelve (12) Pins Scored for your Alliance. (Does not include Pins Scored in Quadrants on the
opposing side of the Autonomous Line.)
2. At least four (4) Goals each contain at least two (2) Pins Scored for your Alliance. (Does not include Goals in
Quadrants on the opposing side of the Autonomous Line).
3. Neither Robot is contacting the Field Perimeter.
4. At least one (1) Robot is at least partially within the Midfield.

---

76
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Rule Modifications: Robot Skills Challenge
All rules apply from V5RC Section 3: Robot Skills Challenge, with no modifications other than those noted
below.
<VURS1> VEX U Robot Skills Matches are set up differently than VEX V5 Robotics Competition Robot Skills
Matches, with the following modification:
a. Modified Match Loads. Four (4) red/yellow Pins, six (6) blue/yellow Pins, and ten (10) Cups begin off the Field
as Match Loads. See figure VURS1.
<VURS2> Teams are permitted to use both Robots in their VEX U Robot Skills Matches, per <VUT1> and
<VUR1>.
<VURS3> Both Robots must start the Robot Skills Match in legal starting positions for the red Alliance. All
other portions of rule <SG1> apply. Each Robot must use one red/yellow Pin as a Preload in accordance with
<SG5>.

---

77
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure VURS1: The Field setup for a VEX U Robot Skills Match.

---

78
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Rule Modifications: Tournament
<VUT1> Instead of a 2-Team Alliance format, VURC Head-to-Head Matches will be played 1-Team vs. 1-Team.
Each Team will use two (2) Robots in each Match.
a. Teams are allowed to build and bring as many Robots as they would like, but only two (2)—one of each size
as described in <VUR1>—may be brought from the pit to the playing Field for any Match.
b. All Robots must pass inspection before they are allowed to compete.
<VUT2> Qualification Matches will be conducted in the same manner as in a V5RC tournament, but in the
revised 1v1 format described in <VUT1>.
<VUT3> Elimination Matches will be conducted in the same manner as in a V5RC tournament, but without
an Alliance Selection. At the end of the competition, one Team will emerge as the Tournament champion.
<VUT4> The Autonomous Period at the beginning of each Head-to-Head Match will be 30 seconds.
a. Human interaction with Robots during the Autonomous Period is strictly prohibited.
b. If both Teams complete their routines before 30 seconds have elapsed, they have the option to signal
that they wish to end the Autonomous Period early. Both Teams and the Head Referee must all agree on
the “early stop.” This is not a requirement, and the option must have been established for all Teams at the
event, such as during the event meeting.
<VUT5> The Driver Controlled Period is shortened to 90 seconds and immediately follows the
Autonomous Period.
<VUT6> VURC tournaments have fewer Teams in Eliminations. The number of Teams in Elimination Matches
for a given event is determined as follows, except in extraordinary circumstances with the permission of VEX
Robotics. A number of Teams below 16 will result in one or more Byes for highest-ranking Teams.
# of Teams # of elimination Teams
16+ 16
<16 # of Teams

---

79
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Rule Modifications: Robot
<VUR1> Teams may use two (2) Robots in each Match.
a. Both Robots may only be built from the following materials:
i. Official VEX Robotics products (see <VUR2>)
ii. Fabricated Parts made by the Team (see <VUR3> through <VUR7>)
iii. Commercially-available springs, fasteners, and bearings (see <VUR8>, <VUR9>, and <VUR15>)
iv. A legal electronics system (see <VUR10> and <VUR11>)
v. Any legal Additional Electronics (see <VUR12>)
vi. A legal pneumatics system (see <VUR14>)
vii. Unmodified legal Raw Stock (see <VUR4> and <VUR5>)
b. One Robot must be no larger than 24” (609.6 mm) x 24” (609.6 mm) x 24” (609.6 mm) at the start of the
Match.
c. One Robot must be no larger than 15” (381 mm) x 15” (381 mm) x 15” (381 mm) at the start of the Match.
<VUR2> Teams may use any official VEX Robotics products, other than the exceptions listed in the tables
below, to construct their Robot. This includes those from the VEXpro, VEX EXP, VEX IQ, VEX GO, VEX 123,
VEX CTE, and VEX Robotics by HEXBUG* product lines. Rule <R28> applies, but most modifications to non-
electrical components are allowed.
SKU Description
217-8080 Talon SRX
217-9191 Victor SPX
217-9090 Victor SP
217-4243 Pneumatic Control Module
217-4244 Power Distribution Panel
217-4245 Voltage Regulator Module
SKU Description
217-4347 775pro
217-2000 CIM Motor
217-3371 Mini CIM Motor
217-3351 BAG Motor
217-6515 Falcon 500
This rule takes precedence over all other rules regarding Raw Stock and/or Fabricated Parts, such as <VUR5>.
* The HEXBUG brand is a registered trademark belonging to Spin Master Corp
As of November 2025, all VEXpro parts are discontinued. To maintain a level playing field
and ensure all VURC Teams have access to the same library of parts, functionally equivalent,
drop-in part substitutes for VEXpro parts may be considered to meet the intent of <VUR2>.
In order to meet our intent, each functionally equivalent, drop-in part must:
a. Match the form, fit, and function of the VEXpro part it is replacing.
b. Not provide a discernible or perceivable advantage over the comparable VEXpro part.
c. Comply with all other applicable VURC Robot rules.

---

80
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Teams should be prepared to demonstrate or defend their part substitutes in inspection as
necessary, when possible.
The Game Design Committee understands that this may be difficult in the immediate future,
as the VEXpro website no longer hosts documentation on VEXpro parts, including but not
limited to part descriptions, drawings, pictures, etc. The Game Design Committee is working
with VEX Robotics to explore and identify a better long-term solution to this problem. In the
interim, we ask all Teams, referees, inspectors, and the entire VURC community to:
1. Work collaboratively, understanding that the intent of this clarification is to allow Robots to
compete with minimal to no modification, not to prevent participation.
2. Apply <G1> and <G3>when interpreting this guidance. We recognize this may temporarily
result in a challenging inspection environment; good faith is essential.
3. Refrain from exploiting this allowance. We intentionally and meaningfully chose to attempt
to maintain product legality and competitive fairness while products are discontinued. De-
liberately pushing boundaries may force the Game Design Committee to entirely reassess
the legality of all VEXpro components.
<VUR3> Fabricated Parts may be made by applying the following manufacturing processes to legal Raw
Stock:
a. Additive manufacturing processes, such as 3D printing.
b. Subtractive manufacturing processes, such as cutting, drilling, routing, or machining.
c. Bending, such as sheet metal braking or thermoforming.
d. Attaching materials to one another, such as welding or chemically bonding (e.g., epoxy).
e. Molding of non-metals, such as injecting polyurethane into a 3D printed mold.
<VUR4> Fabricated Parts must be made from legal Raw Stock. To be considered Raw Stock, the material must
be obtained in one of the following forms before undergoing the fabrication processes listed in <VUR3>:
Type Shape / Profile Examples
1 Sheet Flat Plane
• Sheet metal
• ⅛” polycarbonate sheet
• Plywood
2 Solid Billet 
“Thick” rectangular
beam / block
• 4” x 4” x 6” solid aluminum billet
• 2” x 2” x 2” acetal block
3 Solid Bar “Thin” rectangular beam 
• 2x4 wood planks
• ¼” x 3” aluminum bars
4 Hollow Bar Hollow rectangular beam • 1” x 1”, 1/32” wall aluminum box tube
5 Solid Rod
Cylinder,
Hexagonal or Rounded
Hexagonal Stock
• ¼” steel rod
• ¼” acetal rod
• VEXpro Hex Shaft
6 Hollow Rod / Tube
Hollow Cylinder,
Drilled/Threaded
Hexagonal or Rounded
Hexagonal Stock
• Copper tubing
• PVC pipe
• VEXpro ThunderHex Stock

---

81
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
7 Angle 90° “L” shape • 1” x 1”, 1/16” thickness aluminum angle
8 U- / C-Channel “U” or “C”. • 1/4” High x 1” Wide Aluminum U-Channel
9 
Non-Metal 3D
Printer Filament 
Thin cylinder
• PLA or TPU filament
• Composite nylon filament (e.g.
Markforged OnyxTM)
10 
Synthetic Polymer
used for Molding 
Liquid 
• Polyurethane
• Silicone
11 Solid Sphere 
Solid (not hollow)
uniformly rounded stock
• Steel ball bearing
• Shaped wood finial
Teams are not required to exhaustively define the specific material type for each component
of every Fabricated Part in their Engineering Notebook, as it should be obvious from the engi-
neering drawings required by <VUR7>. However, unusual parts should be expected to receive
increased scrutiny.
If any materials do not easily fall into one of these categories, then that is probably an indica-
tion that it is not intended to be a legal type of Raw Stock. If a Team cannot demonstrate that
the component was made from a legal type of Raw Stock, then they will be asked to remove it
from their Robot.
<VUR5> The following material types are not considered Raw Stock, and are therefore not permitted:
Type Examples
1
Any otherwise-legal Raw Stock that
has been post-processed by drilling,
machining, or otherwise removing material
• Angle aluminum with regularly-
spaced holes or slots
• Perforated sheet metal
2 
Extrusions that do not fall under one
of the categories listed in <VUR4>
• Non-rectangular aluminum extrusions,
such as 80/20, T-slot, or Octanorm
• Gear stock
3
Assembled items (or pre-arranged
kits of unassembled items) that form
a single, more complex component
• Gearboxes
• Claw mechanisms
• Swerve drive modules
4
Commercial Off-the-Shelf items
that are intended to be used
with minimal modification
• Wheels
• Gears
• Timing belts and pulleys
5 
Materials that are intended
to be cast or sintered
• Resin / powdered-bed 3D printing
• Molten aluminum used for sand casting
Note: <VUR2> takes precedence over this rule. Materials purchased from VEX Robotics that fall under
one of these categories (e.g., VersaFrame pre-drilled extrusion) are permitted.
In industry, terms like “raw stock,” “raw material,” and “material stock” are often used inter-
changeably and cover an extremely broad scope of physical goods. The lists in <VUR4> and
<VUR5> are intended to explain what specific material types and profiles fall under the defined
term “Raw Stock” in the context of the VEX U competition.

---

82
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
<VUR6> Fabricated Parts cannot be made from Raw Stock which poses a safety or damage risk to the event,
other Teams, or Field Elements. Examples of prohibited materials include, but are not limited to:
a. Any material intended to produce flames or pyrotechnic effects.
b. Any material that is liquid at the time of the Match. Examples include hydraulic fluids, oils, greases, liquid
mercury, and tire sealant.
i. This does not include fabrication processes that involve the use of liquids, such as milling coolant or
epoxy.
c. Any matter that shatters or otherwise presents an excessive Field/safety hazard upon failure. Examples
include fiberglass, acrylic, and carbon fiber sheet/tube stock.
i. This rule refers specifically to material legality itself. Any potentially unsafe mechanisms made from
legal Raw Stock may still be addressed by <S1> and <R19>.
ii. 3D printer filaments that include carbon fiber (or similar) additives or carbon fiber (or similar) inlay are
exempt from this exception, and are considered legal for use in Fabricated Parts.
<VUR7> Fabricated Parts must be made by Team members. Any Fabricated Parts must be accompanied by
documentation that demonstrates the Team’s design and construction process for that Fabricated Part.
a. The minimum acceptable form of documentation is an engineering drawing with multiple views for the
part in question. These drawings may be included in a Team’s Engineering Notebook or in a standalone
appendix to the Engineering Notebook.
b. Any Fabricated Part must have been entirely designed and produced by Team members. For example, parts
ordered by the Team and 3D printed by a third party would be prohibited.
c. Teams will be required to provide this documentation as requested by inspectors, Head Referees, or judges
at any time at an event. Failure to provide acceptable documentation will result in the part being
deemed illegal for use; therefore, <R2>, <G5>, and/or <G1> will apply.
<VUR8> Teams may use commercially-available springs on their Robots. For the purposes of this rule, a
“spring” is any device used for storing and releasing elastic potential energy. Examples include, but are not
limited to:
a. Compression, tension, torsion, constant force, or conical springs made from spring steel.
b. Springs made from elastic thread or rubber, such as surgical tubing, bungee cords, or stretchable braided
rope.
c. Closed-loop (pneumatic) gas shocks.
Note: Gas shocks are not considered pneumatic devices in the context of <VUR14>. Gas shocks may
not be modified in any way.
<VUR9> Teams may use commercially available fastener hardware on their Robot. Examples include (but are
not limited to):
● Screws, nuts, rivets, and heatset inserts
● Hinges, pins, rod ends, threaded rods, and hose clamps

---

83
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
● Ancillary fastener accessories, such as washers or spacers
● Adhesives such as epoxy, glue, or tape (only when used to join together two parts)
If the primary function of the part is not “fastening”, then <VUR5>, <VUR6>, and/or <VUR7> take precedence
over this rule. Illegal examples include (but are not limited to):
● A prefabricated non-VEX wheel, even though it may technically connect tread to a shaft
● 80/20 extrusion; other items get “fastened to it”, it is not the part doing the “fastening”
● Using grip tape to improve wheel traction
<VUR10> Each Robot must utilize exactly one (1) V5 Robot Brain and at least one (1) V5 Robot Radio
connected to a V5 Controller.
a. Teams must abide by the power rules noted in <R12> and <VUR12d>.
b. Wireless communication between Robots is permitted if using legal V5 Robot Brains and V5 Robot Radios.
No other types of wireless communication protocols (e.g., radio, Bluetooth, Wi-Fi) are permitted.
<VUR11> There is no restriction on the number of V5 Smart Motors, 11W (276-4840) and/or 5.5W (276-
4842), that Robots may use. No other motors, servos, or electronic actuators are permitted, including those
sold by VEX (e.g., the 2-Wire 393 Motor).
Note 1: Rule <R28> still applies in VEX U. Teams may not modify Smart Motors, and must use official/
unmodified gear cartridges.
Note 2: Commercially available pneumatic actuators and pneumatic solenoids are permitted within the
guidelines of <VUR14>.
Note 3: Legal Additional Electronics may include their own motor, servo, or actuator, per <VUR12>.
Note 4: The 55W drivetrain limit specified in <R11> does not apply for VEX U.
<VUR12> There is no restriction on commercially available Sensors, External Processors, or Additional
Electronics that Robots may use for sensing and processing, except as follows:
a. Sensors and External Processors MUST be connected to the V5 Robot Brain via any of the externally
accessible ports (i.e., without any modification to the commercially available electronics). A Sensor may be
connected to an External Processor which then connects to the V5 Robot Brain.
b. Sensors, External Processors, and Additional Electronics CANNOT directly electrically interface with VEX
motors and/or solenoids.
c. Sensors and Additional Electronics may only receive power from any of the following:
i. Directly from the V5 Robot Brain via any externally accessible port.
ii. From an additional lithium ion, lithium iron, or nickel metal hydride battery pack (only one (1) additional
battery can be used for Sensor/processing power). This additional battery pack must operate at a
maximum of less than 13 volts.

---

84
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
iii. Directly from an External Processor
d. Only the V5 Battery can power the V5 Brain.
e. Sensors, External Processors, and/or Additional Electronics which include a low-powered motor as an
integral part of their primary sensing/processing function, such as an External Processor’s cooling fan or a
spinning Sensor, are permissible.
i. Standalone motors which serve no additional sensing or processing functionality (e.g., using a
commercially-available brushless motor in a drivetrain) are not considered legal Additional Electronics,
and would be considered a Violation of <VUR11>.
f. Pneumatic solenoids are the only types of solenoids that are permitted as Additional Electronics. Solenoids
used for any purpose other than opening and closing a pneumatic valve are considered an actuator and
therefore prohibited, per <VUR11>.
g. <R28> still applies in VEX U, Teams may not alter or modify electronic parts from the VEX product lines. It
is legal to rotate the blocks of the V5 Optical Sensor and V5 Distance Sensor relative to their bases, and to
manufacture custom back plates for these sensors.
<VUR13> Commercially available Electromechanical Assemblies are not legal for use on Robots.
a. For the purposes of this rule, any system that integrates Sensors with other mechanical parts that are
fabricated by anyone other than Team members and which serve more use than the basic definition of a
Sensor would be considered an Electromechanical Assembly, and is therefore not legal.
b. Examples may include but are not limited to: odometry pods.
c. Commercially available Sensors with simple plastic housings that do not have any use beyond protecting
internal components and aiding in mounting of the Sensor are not considered Electromechanical
Assemblies.
The intent of this rule is to remind Teams to focus their efforts on integrating custom parts with
the VEX Robotics ecosystem. The VEX U Competition operates within a semi-closed system,
not an open-build system. Teams should make efforts to use VEX Robotics parts where
possible. Parts like additional Sensors (LIDAR, encoders, etc.) should generally be considered
okay, but assemblies/systems from other robotics suppliers that remove the challenge of
systems integration should not be considered a legal part.
<VUR14> Teams may utilize an unlimited amount of the following commercially available pneumatic
components: cylinders, actuators, valves, gauges, storage tanks, regulators, manifolds, tubing, and solenoids.
a. Pneumatic devices may only be charged to a maximum of 100 psi.
b. Compressors or any other forms of “on-Robot” charging are not permitted.
c. All commercial components must be rated for 100 psi or higher. Teams should be prepared to provide
documentation that verifies these ratings to inspectors if requested.
d. Components must not be modified from their original state, other than the following exceptions:
i. Cutting pneumatic tubing or wiring to length; assembling components using pre-existing threads,
brackets, or fittings; or minor cosmetic labels

---

VEX V5 Robotics Competition Override - Game Manual
85
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
e. If commercially available 12V solenoids are used, these are considered Additional Electronics and must
therefore satisfy all conditions listed in <VUR12>. 12V solenoids may be either powered by an additional
power source (per <VUR12d>), or by a 5V-12V step-up converter from the V5 Robot Brain. If an external
power source (or other Additional Electronics device) is used to interface with the solenoid, Teams MUST
be able to demonstrate that there is no way for the solenoid to receive power while the Robot is receiving a
Disabled state from the field controller.
<VUR15> Teams may use commercially available bearings on their Robot. For the purpose of this rule, a
‘bearing’ is a part that supports external loads, reduces friction, and improves efficiency by facilitating smooth
dynamic motion between components. Legal examples include (but are not limited to):
● Parts supporting rotational motion: radial bearings, roller bearings, thrust bearings, needle bearings,
one-way bearings, bushings, etc.
● Parts supporting linear motion: linear bearings, linear slides, drawer slides, etc.
# Team Composition
We want to see Universities face off in a global head-to-head competition. Schools are not limited to one
Team, and a Team may consist of multiple colleges, but we hope that each Team identifies with and proudly
represents one (1) post-secondary institution. (e.g., “Clarkson University” vs. “UC Santa Barbara”). Of course,
college-level “club” Teams and mixed composition Teams are encouraged to join! However, as noted in
<VUT6>, Students who have not yet graduated secondary school are not eligible to participate in VEX U, even
if they are “dual-enrolled” or taking post-secondary courses. A Student cannot be a member of a V5RC and a
VURC Team simultaneously, regardless of eligibility.

---

Copyright 2026, VEX Robotics, Inc.
## Appendix A
## Field Overview and Specifications

---

A2
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Game Field Introduction
# Appendix A - Field Overview
# Field Overview
This document will provide Bill of Materials (BOM) information and detailed specifications for the Official
Competition Field.
Please note: this Field can utilize both the VEX Portable Competition Field Perimeter (276-8242 and the VEX
Competition Field Perimeter (278-1501) developed by VEX Robotics. Instructions and specifications for these
Field Perimeters are available in separate documents and are important for the Field assembly.
This document is divided up into three sections:
1. Field Overview
2. Field BOM
3. Field Specifications
There is also an accompanying STEP file which can be imported into most 3D modeling programs (e.g.,
Inventor, Sketchup, Solidworks, etc.). This 3D model shows the “official” setup of a VEX V5 Robotics
Competition Override competition Field, as well as detailed models of individual Field Elements. For additional
game-play detail, please refer to the VEX V5 Robotics Competition Override Game Manual.
V5RC Override is played on a 12ft x 12ft foam mat, surrounded by a Field Perimeter, with nine (9) Goals, four (4)
Loaders, and four (4) Toggles on the Field.
The V5RC Override Field consists of fifty-six (56) Cups and sixty-three (63) Pins.
For more details and specific gameplay rules, please refer to the V5RC Override Game Manual.

---

A3
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Game Objects & Field Bill of Materials
All of these items are available for purchase from www.vexrobotics.com
Generic Field Elements - Reusable Each Year
Part Number Description
278-1501 Field Perimeter Frame & Hardware
276-8242 Portable Competition Field Perimeter
276-6905 Anti-Static Field Tiles (18-Pack)
276-7741 Smart Field Controller Kit
Official VEX V5 Robotics Competition Override Specific Elements
Part Number Description Quantity per Full Field
276-9250 V5RC 2026-27 Full Field & Game Element Kit
276-9251 V5RC 2026-27 Game Element Kit 1
276-9252 V5RC 2026-27 Field Element Kit 1 1
276-9091 V5RC Field Element Plates (4-Pack)* 2
*Optional. Only needed if Field Element Plates are not already owned.
Practice Elements
Part Number Description
276-9255 V5RC 2026-27 Scoring Element Kit

---

A4
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Field Specifications Introduction
This section will outline the specifications that are most important to Teams designing a Robot to compete in
VEX V5 Robotics Competition Override. Though many of the critical dimensions are included in this section,
it may be necessary to consult the separate assembly guide and 3D CAD models of the Field for an additional
level of detail. If you can’t find a dimension in the specifications, we include a full model of the Field to “virtually”
measure whatever dimension is necessary.
Field components may vary slightly from event to event. This is to be expected; Teams will need to adapt
accordingly. It is good design practice to create mechanisms capable of accommodating variances in the Field
and Scoring Objects.
Note: Minor Field repairs are permissible, provided that the repairs do not affect gameplay. Examples
of minor Field repairs include (but are not limited to) threadlocker applied to Field Element mounting
hardware. Be sure to check the Official Q&A for specific examples or to get an official clarification.

---

A5
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A6
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A7
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A8
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A9
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A10
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A11
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A12
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A13
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A14
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A15
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A16
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.

---

A17
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Click here to download this .STEP file

---

Copyright 2026, VEX Robotics, Inc.
## Appendix B
## Glossary of Terms

---

B2
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Adult - Anyone who is not a Student or another defined term (e.g., Head Referee).
Alliance - A pre-assigned grouping of two Robots that are paired together during a given Head-to-Head Match.
Alliance Captain - One of the Teams with the privilege of inviting another available Team to form an Alliance for
Elimination Matches. See <T16>.
Alliance Selection - The process of choosing the permanent Alliances for Elimination Matches. The Alliance
Selection proceeds as follows:
1. The highest-ranked Team at the end of Qualification Matches becomes the first Alliance Captain.
2. The Alliance Captain invites another Team to join their Alliance.
3. The invited Team representative either accepts or declines as outlined in <T16>.
4. The next-highest-ranked Team becomes the next Alliance Captain.
5. Alliance Captains continue to select their Alliances in this order until all Alliances are formed for Elimination
Matches.
Alliance Station - The designated regions where the Drive Team Members must remain for the duration of the
Match.
Autonomous Bonus - A point bonus awarded to the Alliance that has earned the most points at the end of the
Autonomous Period. See <SC7> for more information.
Autonomous Coding Skills Match - see Match.
Autonomous Line - The pair of white tape lines that run diagonally across the Field and around the Midfield,
and the space between those lines. See <SG7> for more information.
Autonomous Period - A time period during which Robots operate and react only to sensor inputs and pre-pro-
grammed commands.
Autonomous Points (AP) - The second basis of ranking Teams. An Alliance who wins the Autonomous Bonus
during a Qualification Match earns ten (10) Autonomous Points. In the event of a tie, both Alliances will receive
five (5) Autonomous Points.
# Appendix B - Glossary of Terms

---

B3
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Autonomous Win Point (AWP) - An additional Win Point awarded to any Alliance that has completed a defined
set of tasks at the end of the Autonomous Period of a Qualification Match. Both Alliances can earn an Autono-
mous Win Point if both Alliances accomplish these tasks. See <SC7> for more information.
Builder - Any Student Team member who helps build the Robot. Adults are permitted to teach Builders associ-
ated concepts, but should never work on the Robot.
Bye - A situation in which an Alliance automatically advances to the next round of Tournament play without
competing.
Coder - Any Student Team member who contributes to the computer code that is downloaded onto the Robot.
Adults are permitted to teach Coders associated concepts, but should never work on the code that goes on
the Robot.
Cup - A Scoring Object, measuring approximately 3.15” (80mm) in diameter and 6.5” (164.5mm) tall. Each Cup
consists of two halves: one transparent and one opaque.
Figure C-1: A Cup
Defensive - A category of strategies, Robot actions, and/or Robot statuses that can be employed by a Team
during a Match; see rules <GG14> and <GG15> for more information. A Robot is Defensive while it is engaged in
actions that cannot increase its Alliance’s score for the current Match, and instead limits an opponent’s ability
to score or play the game. A Robot can be in Possession of a Scoring Object and capable of scoring, but still be
Defensive based on its actions. Examples include, but are not limited to:
● De-scoring in a way that doesn’t increase points for the Robot’s own Alliance
● Limiting access to a portion of the Field while not attempting to score
● Holding, blocking, impeding, or otherwise restricting or controlling an opponent’s movements

---

B4
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Remember, Defensive Robot actions or Robot statuses are not automatically Violations.
However, Robot actions or statuses that are performed or achieved in a Defensive manner
are more likely to be Violations, and Teams should be more careful when employing Defensive
strategies.
Designer - Any Student Team member who helps design the Robot to be built for competition. Adults are
permitted to teach Designers associated concepts, but should never work on the design of the Robot.
Disablement - A penalty applied to a Team for a safety Violation. A Team that receives a Disablement is not
allowed to operate their Robot for the remainder of the Match, and the Drive Team Member(s) will be asked to
place their controller(s) on the ground or another safe location outside of the Field, as directed by the Head
Referee.
Disqualification - A penalty applied to a Team for a Major Violation (see <GG6> for more details). If a Team
receives a Disqualification in a Match, the Head Referee will notify the Team of their Violation at the end of the
Match. A Team that receives a Disqualification in a Qualification Match receives zero Win Points, zero Auton-
omous Win Points, zero Autonomous Points, and zero Strength of Schedule Points. When a Team receives a
Disqualification in an Elimination Match, the entire Alliance is Disqualified and they receive a loss for the Match.
At the Head Referee’s discretion, repeated Violations and/or Disqualifications for a single Team may lead to
its Disqualification for the entire Tournament (see <GG6>). A Team that receives a Disqualification in a Driving
Skills Match or Autonomous Coding Skills Match receives a score of zero for that Robot Skills Match.
Drive Team Member - A Student who stands in the Alliance Station during a Match. Adults are not allowed to
be Drive Team Members. See rule <GG1>.
Driver Controlled Period - A time period during which Drive Team Members operate their Robot using a VEX
V5 controller.
Driving Skills Match - see Match.
Elimination Bracket - A schedule of Elimination Matches for eight (8) to sixteen (16) Alliances. See <T17>.
Elimination Match - see Match.
Endgame - A time period consisting of the last 10 seconds of a Head-to-Head Match, in which Robots attempt
to end the Match in the Midfield. See rule <SG12>.
Entanglement - A Robot status. A Robot is Entangled if it has grabbed, hooked, or attached to an opposing
Robot or a Field Element. See rule <GG14>.

---

B5
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Event Partner - The volunteer VEX V5 Robotics Competition Tournament coordinator who serves as an overall
manager for the volunteers, venue, event materials, and all other event considerations.
Field - The entire playing Field, comprising the Floor and the Field Perimeter.
Field Element - The Field, white tape, Loaders, Goals, Toggles, and all supporting structures and accessories
(such as field monitors, etc.).
Field Perimeter - The outer part of the Field, made up of 12 straight sections.
Floor - The interior flat part of the playing Field, made up of an array of six (6) gray foam field tiles wide by six (6)
gray foam field tiles long (totaling 36 Field tiles) that are within the Field Perimeter.
Game Design Committee (GDC) - The creators of Override, and authors of this Game Manual. The Game
Design Committee is the only official source for rules clarifications and Q&A responses; see Section 1.
Goal - One of the nine (9) designated locations around the Field in which Robots attempt to score Pins and
Cups. Goals are octagonal, and each Goal is one of three colors: red, blue, or black. The center Goal is 8.7”
(222.7mm) tall, neutral Goals in Quadrants are 5.8” (146.5mm) tall, and Alliance-specific Goals are 3.25”
(82.5mm) tall.
Figure G-1: The four types of Goals in a V5RC Override Match
Head Referee - A certified impartial volunteer responsible for enforcing the rules in this manual as written.
Head Referees are the only people who may discuss ruling interpretations or scoring questions with Teams
at an event. Large events (e.g., Signature Events, World Championships, etc.) might include multiple Head
Referees at the Event Partner’s discretion.
Head-to-Head Match - see Match.
Holding - A Robot status; see rule <GG17> for more information. Holding is legal until it exceeds the limits in
<GG17>. A Robot is considered to be Holding if it meets any of the following criteria during a Match:

---

B6
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
● Trapping - Limiting the movement of an opponent Robot to a small or confined area of the Field, approx-
imately the size of one foam field tile or less, without an avenue for escape. Note that if a Robot is not
attempting to escape, it is not considered Trapped.
● Pinning - Preventing the movement of an opponent Robot through contact with the Field Perimeter, a Field
Element, or another Robot.
● Lifting - Controlling an opponent’s movements by raising or tilting the opponent’s Robot off of the Floor.
Preventing a Robot that is already off of the Floor from returning to the Floor may also be considered Lifting
or Trapping.
Loader - One of the four designated locations (two per Alliance) around the Field where Drive Team Members
can introduce Match Load Pins and Cups. See rule <SG11>.
Figure L-1: A Loader
Load Zone - An infinitely tall three-dimensional volume bounded by the inner edges of the Field Perimeter
and the outer edges of the red or blue tape lines surrounding each Loader.
Match - A set time period, consisting of an Autonomous Period and/or Driver Controlled Periods, during
which Teams play a defined version of Override to earn points.
Match types:
● Autonomous Coding Skills Match - A Robot Skills Match in which a single Robot operates during a one
minute Autonomous Period. There is no Driver Controlled Period. Teams can elect to end an Autonomous
Coding Skills Match early if they wish to record a Skills Stop Time.
● Driving Skills Match - A Robot Skills Match in which one Team operates their Robot during a one minute
Driver Controlled Period. There is no Autonomous Period. Teams can elect to end a Driving Skills Match
early as described in rule <RSC5> if they wish to record a Skills Stop Time.
● Elimination Match - A Head-to-Head Match used in the process of determining the champion Alliance.
Alliances of two (2) Teams face off according to the Elimination Bracket; the winning Alliance moves on to
the next round.

---

B7
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
● Head-to-Head Match - A Match that consists of two Alliances that work to outscore the other Alliance in a
two-minute Match. Qualification Matches, Finals Matches, and optional Practice Matches are Head-to-Head
Matches.
● Practice Match - A non-scored Head-to-Head Match used to provide time for Teams to get acquainted with
the official playing Field and procedures. Head Referees should not record or track standard gameplay
Violations that occur during Practice Matches. Egregious Violations may be recorded and tracked at the
discretion of the Head Referee.
● Qualification Match - A Head-to-Head Match that is used to determine Teams’ rankings for Alliance Selec-
tion. Each Qualification Match consists of two Alliances competing to earn Win Points, Autonomous Points,
and Strength of Schedule Points.
● Robot Skills Match - A Driving Skills Match or Autonomous Coding Skills Match.
Match Type Participants Specific Rules 
Autonomous
Period (m:ss)
Driver Controlled
Period (m:ss)
Head-to-Head Match
Two Alliances
(red/blue) each
composed of
two Teams, with
one Robot each
Scoring (“SC”),
General Game (“GG”)
and Specific Game
(“SG”) sections
0:15 1:45
Driving Skills Match 
One Team, with
one Robot 
Section 3 None 1:00
Autonomous Coding
Skills Match
One Team, with
one Robot 
Section 3 1:00 None
VEX U Robotics
Competition
Head-to-Head
Match
Two Teams (red/
blue), with two
Robots each
Section 6 0:30 1:30
VEX U Robotics
Competition Driving
Skills Match
One Team, with
two Robots 
Section 6 None 1:00
VEX U Robotics
Competition
Autonomous
Coding Skills Match
One Team, with
two Robots 
Section 6 1:00 None
Match Load - Scoring Objects that begin the Match in an Alliance Station, which may be introduced to the Field
by Drive Team Members during the Match. See rule <SG11>.
Match Schedule - A list of Matches that is generated at the start of an event. The Match Schedule includes the
predetermined, randomly-paired Alliances that will be competing in each Qualification Match, and the expected
start times for these Matches. The Match Schedule may be subject to change at the Event Partner’s discretion.

---

B8
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Midfield - The square-shaped central area of the Field in which Robots attempt to end the Match to score
additional points. The Midfield is an infinitely tall three-dimensional volume bounded by the inner edges of the
white tape line square, and can be entered by all Robots during the Autonomous Period. See Figure Q-1.
Notebooker - Any Student Team member who contributes to the Team’s engineering notebook or associated
documentation. Adults are permitted to teach Notebookers associated concepts, but should never work on the
engineering notebook or other documentation.
Offensive - A category of strategies, Robot actions, and/or Robot statuses that can be employed by a Team
during a Match; see rules <GG14> and <GG15> for more information. A Robot is Offensive while it is engaged
in actions that could directly increase its Alliance’s score for the current Match. Examples include, but are not
limited to:
● Adding a Scoring Object to a Goal to score points
● Moving toward a Goal with a Scoring Object that could earn points for their Alliance
● Changing the status of a Toggle
● Achieving (or attempting to achieve) any Robot status that adds points to their Alliance’s score
● Obtaining (or attempting to obtain) Scoring Objects
Owned - A yellow Pin status. A Placed yellow Pin is Owned by an Alliance if the Toggle in that Quadrant is set to
the Alliance’s color.
Placed - A Scoring Object status. See <SC2>.
Pin - A Scoring Object measuring approximately 1.6” (40mm) in diameter and 6.5” (165mm) tall . Each Pin
consists of two halves, and each half is red, blue, or yellow.
Figure MS-1: An example of a Qualification Match Schedule

---

B9
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Figure P-1: The four types of Pins in a V5RC Override Match
Possession - A Robot / Scoring Object status. A Scoring Object is considered Possessed by a Robot if the
Scoring Object is fully supported by the Robot, or if the Robot’s movement in all directions would result in
maintained control of the Scoring Object (i.e., the Scoring Object moves along with the Robot as the Robot
turns and moves forward, backward, left, and right).
Example 1 (Possession): A Scoring Object held by a Robot’s intake, either in the air or along the
Floor, would be considered Possessed since the Scoring Object would move along with the
Robot as the Robot turns and moves in all directions.
Example 2 (No Possession): Scoring Objects being pushed forward on the Floor by the front
face of a Robot would not be considered Possessed by that Robot, since driving backward
would cause the Robot to lose control of those Scoring Objects (i.e., the Scoring Objects would
not move backward with the Robot, and would become separated from it). This remains true
whether the Robot’s face is flat, convex, or concave, as long as movement in at least one direc-
tion would result in the Robot losing control of the Scoring Object(s).
Practice Match - see Match.
Preload - The Pins, one (1) per Robot, placed by each Team prior to the start of each Match. See <SG5>.
Quadrant - One of four designated triangular areas of the Field. Each Quadrant contains two Goals and one
Toggle. Robots may score Pins in the Goals within a Quadrant and may attempt to set that Quadrant’s Toggle to
control ownership of yellow Pins Scored in that Quadrant.
A Quadrant is defined by the outer edges of the white tape lines, the Field Perimeter, and the colored tape that
defines the Load Zones.
Each Quadrant is described as red or blue based on the Alliance-colored Goal it includes. See Figure Q-1.

---

B10
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Qualification Match - see Match.
Robot - A machine that has passed inspection, designed by Student Team members to execute one or more
tasks autonomously and/or by remote control from a Drive Team Member.
Robot Skills Match - see Match.
Scored - A Scoring Object status. See <SC3>.
Scorekeeper Referee - An impartial volunteer responsible for tallying scores at the end of a Match. Score-
keeper Referees do not make ruling interpretations, and should redirect any Team questions regarding rules
or scores to a Head Referee.
Scoring Object - A Cup or Pin.
Skills Stop Time - The time remaining in a Robot Skills Match when a Team ends the Match early. See <RSC5>
for more details.
Strategist - Any Student Team member who contributes to the Match strategies used to score points during
a Qualification Match or Robot Skills Match, including assessing the impact of other Teams’ performance
and strategies on the Team’s strategy (e.g., scouting). Adults are permitted to teach Strategists associated
concepts, but should never create or dictate a Team’s Match strategy.
Figure Q-1: An overhead view of the Override Field with the four Quadrants (red and blue) and Midfield (yellow) highlighted.
The Quadrant labels (Red 1, Red 2, Blue 1, and Blue 2) correspond to scoring input areas inside Tournament Manager, and are not relevant
outside of the Tournament Manager software.

---

B11
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Strength of Schedule Points (SP) - The third basis of ranking Teams. Strength of Schedule Points are equiv-
alent to the score of the losing Alliance in a Qualification Match. In the event of a tie, both Alliances receive
Strength of Schedule Points equal to the tie score. If both Teams on an Alliance are Disqualified, the Teams on
the not Disqualified Alliance will receive their own score as Strength of Schedule Points for that Match.
Student - A person that meets the Student Eligibility Policy requirements for their respective competition
program and level.
Team - One or more Students make up a Team. In the context of this game manual, Student Team members fill
multiple roles related to Robot design, build, coding, strategy, and documentation. See <G2>, <G4>, <G5> for
more information. Adults may not fulfill any of these roles. See Appendix D for more information about Team
classifications and Student roles.
Time-out - A single break period no greater than three minutes (3:00) allotted for each Alliance during the
Elimination Bracket. See <GG7>.
Toggle - One of four (4) triangular shaped Field Elements mounted to the Field Perimeter that can be Owned
to control yellow Pins Scored in the corresponding Quadrant. Each Toggle has 3 sides that, when viewed from
inside the Field, indicate which Alliance Owns the Toggle. Toggles are 25.8” (656.2mm) long and each face of
the triangle is approximately 2.05” (52.2mm) wide . See <SC4> for more information.
Figure T-1: A Toggle.
Tournament - A competition event that includes scored Matches, and which is run by an Event Partner.
Violation - The act of breaking a rule in the game manual. See Appendix C for additional information on Viola-
tions and penalties.
● Minor Violation - A Violation which does not result in a Disqualification.
● Major Violation - A Violation which results in a Disqualification.
● Match Affecting - A Violation which changes the winning and losing Alliance in the Match.

---

VEX V5 Robotics Competition Override - Game Manual
B12
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Win Percentage (WP) - Replaces Win Points in a league event. Win Percentage is calculated by the number of
wins divided by the number of Qualification Matches the Team plays. In cases of a tie, the Team is given a 0.5
number of “wins” for that Match. The Autonomous Win Point is also considered 0.5 “wins,” added to the total
number of wins.
Win Points (WP) - The first basis of ranking Teams. Teams will receive zero (0), one (1), two (2), or three (3) Win
Points for each Qualification Match. Unless a Team is Disqualified, both Teams on an Alliance always earn the
same number of Win Points.
● One (1) Win Point is awarded for completing the Autonomous Win Point task(s).
● Two (2) Win Points are awarded for winning a Qualification Match.
● One (1) Win Point is awarded for tying a Qualification Match.
● Zero (0) Win Points are awarded for losing a Qualification Match.

---

Copyright 2026, VEX Robotics, Inc.
## Appendix C
## Rule Violations

---

C2
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
This appendix is intended to provide additional guidance regarding rule Violations within the VEX V5 Robotics
Competition, and offer further explanation on how Violations may be interpreted and enforced during
Matches. It is designed to promote consistency in officiating and to help Teams better understand how actions
on the Field may impact scoring, Match outcomes, and referee decisions.
This appendix does not supersede any existing rule, but instead serves as a secondary resource to aid in their
application. If no Violation Notes are found in a given rule, then it should be assumed that the below “default”
definitions apply.
Violation - The act of breaking a rule in the Game Manual.
● Minor Violation - A Violation which does not result in a Disqualification.
० Accidental, momentary, or otherwise non Match Affecting Violations are usually Minor Violations.
० Minor Violations usually result in a verbal notification from the Head Referee during the Match, which
should serve to inform the Team that a rule is being Violated before it escalates to a Major Violation.
● Major Violation - A Violation which results in a Disqualification.
० Unless otherwise noted in a rule, all Match Affecting Violations are Major Violations.
० If noted in the rule, egregious or strategic Violations or intentional actions that result in Violations may
also be Major Violations.
० Multiple Minor Violations within a Match or Tournament may escalate to a Major Violation at the Head
Referee’s discretion or as specified in a rule. Minor Violations carry over into Elimination Matches
unless otherwise specified within a rule.
● Match Affecting - A Violation which changes the winning and losing Alliance in the Match.
० Multiple Violations within a Match can cumulatively become Match Affecting.
० When evaluating if a Violation was Match Affecting, Head Referees will focus primarily on any Robot
actions that were directly related to the Violation.
० Determining whether a Violation was Match Affecting can only be done once the Match is complete and
the scores have been calculated.
० To determine whether a Violation may have been Match Affecting, check whether the Team who com-
mitted the Violation won or lost the Match. If they did not win the Match, then the Violation could not
have been Match Affecting, and it was very likely a Minor Violation.
# Appendix C - Rule Violations

---

Copyright 2026, VEX Robotics, Inc.
## Appendix D
## Team Classifications and Student Roles

---

D2
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
# Appendix D - Team Classifications and Student Roles
One or more Students make up a Team. To participate in an official VEX V5 Robotics Competition event, a Team
must first register on events.vex.com and receive a VEX V5 Robotics Competition Team number. A Team’s
unique number identifies their organization and their Team within that organization. Each Team must design
and build their own Robot, create their own code, develop their own strategies to play the game, and maintain
their own engineering notebook if they choose to use one.
● A Team is classified as a Middle School Team if all members are Middle School Students.
● A Team is classified as a High School Team if any of its members are High School Students, or if the Team
is made up of Middle School Students who declare themselves “playing up” as High School Students by
registering their Team as a High School Team.
● Once a Team has competed in an event as a High School Team, that Team may not change back to a Middle
School Team for the remainder of the season. If a Team mistakenly registers as a Middle School Team but
is ineligible for that age group, their registration may be revised mid-season with the assistance of VEX
Robotics; all prior qualifications for the season will be lost.
● Teams may be associated with schools, community/youth organizations, or groups of neighborhood
Students.
In the context of this Game Manual, Student Team members fill multiple roles related to Robot design, build,
coding, strategy, and documentation. See <G2>, <G4>, <G5> for more information. Adults may not fulfill any of
these roles.
● Designer - Any Student Team member who helps design the Robot to be built for competition. Adults are
permitted to teach Designers associated concepts, but should never work on the design of the Robot.
● Builder - Any Student Team member who helps build the Robot. Adults are permitted to teach Builders
associated concepts, but should never work on the Robot.
● Coder - Any Student Team member who contributes to the computer code that is downloaded onto the
Robot. Adults are permitted to teach Coders associated concepts, but should never work on the code that
goes on the Robot.
● Strategist - Any Student Team member who contributes to the Match strategies used to score points
during a Qualification Match or Robot Skills Match, including assessing the impact of other Teams’ per-
formance and strategies on the Team’s strategy (e.g., scouting). Adults are permitted to teach Strategists
associated concepts, but should never create or dictate a Team’s Match strategy.
● Notebooker - Any Student Team member who contributes to the Team’s engineering notebook or asso-
ciated documentation. Adults are permitted to teach Notebookers associated concepts, but should never
work on the engineering notebook or other documentation.

---

D3
VEX V5 Robotics Competition Override - Game Manual
Version 2.0 - Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc.
Copyright 2026, Innovation First, Inc./ VEX Robotics, Inc. Version 1.1 - Released August 6, 2026
Unauthorized copying, reproduction, adaptation, or use of this material in whole or in part
without the express prior written consent of Innovation First, Inc. is strictly prohibited.
Unauthorized use, including use for or in connection with any non-VEX event or competing
competition or program, is subject to civil and/or criminal prosecution of the responsible
person(s).