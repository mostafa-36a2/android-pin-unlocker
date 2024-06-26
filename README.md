()[https://youtu.be/4QeQcnztk2Y]
[![YouTube version of the README](https://img.youtube.com/vi/4QeQcnztk2Y/0.jpg)](https://youtu.be/4QeQcnztk2Y)

### Story of phone lock
- I was trying to have a phone detox on Eid (Vacation of one week), And a strange/irrational Idea was fired.
- To lock the phone with a random pin, so I can forget it now, and remember it later when I focus.
- After 1 hour, I was trying to remember the password, with no success.
- Thankfully, Infinix allows a pin trial each 30 seconds, with no limits. 
### Manual trials
I was trying manually, writing down what pins I am checking
#### What I can remember
- I had a vague memory of what parts of the numpad I was pressing, not numbers, but areas.
- I remember touching 3 or 6 in the middle so I trick my self into different possibilities.
- I remember that it was the 3rd number, but I was not totally sure.
- Each time I question my memory, it got more shaded.
#### Manually trying +700 times
It was stupid, just did a lot of manual trials, tracking with pencil and paper, I have reached +700 manual trial, unfortunately, all of my high priority tries was wrong.

Tracking was an essential part, if no tracking, I can go in loops of repeating the same pin code several times.
I did it completely on paper, then I have tried Excel sheet. 

Long story short, manual testing is not feasible:
- Error prone
- Time consuming
- No logs, you can't even trust the manual tracking because it is also error prone.
### Automation
Automation should always have a reason, don't just automate for fanciness.
But after trying manually, and figure our the cons of that, automation was the only solution.

Several ideas were around:
- Android unlock, tried with [[Aziz]] to do some Android hacking mode, but the memory was ciphered.
- What about a Robotic arm, let's buy a Lego kit and learn from scratch! gracefully, Aziz canceled that Idea.
- OTG mouse! seems not working on my phone.
- OTG Keyboard! that last chance
#### OTG Keyboard
We have several assumptions that we have to check before go with this solution:
- Do the OTG keyboard work when the phone is locked?
- What USB Type?
- How to control the numpad with keyboard?
#### Automate with computer
- I need to link my computer to the phone, so I can test all the possibilities. 
- Aziz, came to the rescue with [[ESP32]] as OTG solution, [espusb](https://github.com/cnlohr/espusb) repo, and [this](https://www.youtube.com/watch?v=ntm1iTQdCzE) YouTube tutorial.
- First we were trying with the user interface.
![[doc/assets/3-access-webui.png]]

![[doc/assets/unlock-phone-open-loop.jpg]]
- I had made before a prioritization list with some excel formulas, now I am ready with the possibilities. 
#### Open loop design
Started with an open loop system.
![[doc/assets/open-flow.jpg]]
But could't trust it, it was unreliable due to several reasons:
- No feedback: The code can't find out the result of orders.
- Uncertain with the results: I can't trust the code as there is no feedback, what if it was dropping some possibilities.
- No logs: There was a lack of logs and tracking for code actions.
- Used time gaps: I was depending on `sleep` after each order, to give the system the needed time to respond, but it was unreliable, and it causes timeouts. 
- Monitor the bad captures manually: I was getting one snapshot with camera after each loop. manually checking the results.
##### Sources of unreliability
- Bread board connectivity
- ESP timeouts
- web socket issues (Due to the lack of experience)
- Lossy OTG connection
- Phone response issues (sometimes, it hangs for milliseconds)
#### Close loop design

- To close the loop, we need to capture images from the web cam, with some [[cv2]] processing, to find out where is the current highlighted element.
- Then we want to capture each number entry.
- Manually monitoring the snapshots, just to check that the process is going correctly.
- Like any computer vision project, several issues raised as illumination and noise. 
#### Handling failed attempts
After manually checking the results, we have found several failed attempts, decided to manually test them.
But wait, there were few failed attempts, less than 1%, what is the possibility that the system failed on the correct pin code! it should be a very low possibility, so I have decided to skip them until the end.
#### Final results
- After several days, of keeping the code running at night, checking on morning.
- I came in one day,to find the phone shutdown, and the script stopped.
- Checked the snapshot, and find this.
![[Pasted image 20240626153827.png]]
Finally unlocked 2476

----

### Lessons learnt
#### Personal
- Your memory can be manipulated easily, don't trust it
- Uncertainty is a killer (System Reliability is hard) fear of failure was killing.
- Friends (With high quality expertise) are saviors
- Motivation and challenges are the driver of your actions, I would never attempt such a project if I was not forced to.
- Hey programmers, try Arduino and hardware projects.
#### System design
- Human manual work is the base, don't underestimate.
- Prefer closed loop system always for reliability
- Logs are essentials, you cannot trust a system with no logs
- Prioritizing chances on random brute force
- Documentation is an accomplishment, I wished that I have documented more during the process. 
#### Phones
- 4 Digits pin is insecure testing.   
- Keeping phone away is not impossible
- Phones are dangerous, I was going to be kicked off my WhatsApp business, my gmail needs [[2FA]].