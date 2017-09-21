---
layout: post
published: true
title: Spam Alert! Machine Learning is filtering your emails.
image: >-
  https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAepAAAAJDM3YjY5NjJiLTkyZmEtNDY1OS05ZDUxLWI4NmQ3MTc0MzA2MQ.jpg
---
*"Congratulations, you have won a lottery worth 10 million $ !! Reply with your credit card details to avail this."*

No, I'm not offering any lottery, this is just an example of a spam e-mail. Nowadays we don't bother much about these emails because they automatically land in our spam folders. But 20 years back, this was not the case. So let's take a seat on a time machine and ride back to the past.

## The 1990s - Just send it!

The 1990s marked the beginning of internet with Hotmail being the first web-based email provider. Standards such as sender authentication, whitelisting, etc. were unknown. Marketers exploited this opportunity. The strategy was 'just send it'. As a result, our inboxes get flooded with junk emails every day.

*Can you imagine wasting 6 hours every day just to clean your inbox?*

It was after this era when Spam Filters came to our rescue! Email providers started filtering emails based on sender's identity, analysing message texts, etc.

Let's understand the basic idea behind spam filters.

## What does a spam filter do?

![Spam filter working](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAepAAAAJDM3YjY5NjJiLTkyZmEtNDY1OS05ZDUxLWI4NmQ3MTc0MzA2MQ.jpg "Spam Filter")

* Email comes from various senders and organizations.
* Spam filter moves legitimate emails to inbox and rest of them to the spam folder.

## How does it work?

Spam filter uses Machine Learning techniques to filter an email. It looks for several features in an email, based on which it decides whether an email is a spam or a ham(term used for legit emails).

**1. Where the message came from?**

Spam filter analyses the sender's address. Based on this, fraud emails can be detected. An anti-spam filter SpamAssasin tracks the network of the message source, maintains a list and looks it up in several other lists. Each time the network appears in a list, the spam score of the message is increased a little.

**2. Which software sent the message?**

Most legit mail comes from big email providers such as Gmail, Outlook(Hotmail), Yahoo, etc. Spam, however, is distributed by a software that is designed to send out millions of messages as quickly as possible. The spammer does not want the messages to be easily traced back to their source. SpamAssassin looks for clues in the message headers that indicate that the message was sent by a spam engine rather than a real mailer.

**3. What the message seems like?**

Spam Filter also looks at the subject and the body of the message just as a person understands that a message "looks like a spam". It searches for strings like "lottery", "buy now", "lowest prices", "click here", etc. It also looks for flashy HTML such as large fonts, blinking text, bright colours, etc. Many spam filters compare the amount of suspicious text to the total amount of text so that an entire 12-page paper doesn't get blocked based on a few suspicious words.

## Challenges to spam filters

Spam filters although a very helpful tool, poses a few challenges. Wondering how?

**False Positives-**

Let's say someone sent you an important email and it went unnoticed because it landed in your spam folder. Maybe you could miss some important opportunity, invitation or anything just because of a fault in the probabilistic methods. In machine learning terminology, this case is called False Positive (i.e. a test result which wrongly indicates that a particular condition or attribute is present).

**Tweaks by Spammers-**

Another challenge is to face spammers who try to trick the filter by modifying emails in such a way that it could not be detected. For example, if a spam filter is made to look for the word **'buy now'** then spammers modify the text like **'B-U-Y N-O-W'**, **'buy noww'**, etc. to surpass the filter.

## How is Google addressing these challenges?

In February 2012, Microsoft boasted that its spam filters were removing all but 3% of the junk emails from Hotmail. Google responded by claiming that Gmail removes all but 1% of spams, adding that its false positive rate is also about 1%.

The relative success of both these companies showed that machine learning technologies were working. But 1% spam is still pretty annoying. And 1% false positive rate is, well, even more annoying.

Naturally, these companies keep improving their spam filtering techniques. Now, Google has come up with a new set of machine learning tools based on neural networks and deep learning. It has decreased their spam rate down to 0.1 percent, and false positive rate down to 0.05%.

“One of the great things about machine learning is that it adapts to changing situations.” says John Rae-Grant, a senior product manager for Gmail.

## End Notes

The spam filter is no doubt an important tool in the web mailing world. And, companies like Google, Microsoft make sure that they don't just filter junk mails based on pre-existing tools but also create their methods using advanced machine learning techniques.
