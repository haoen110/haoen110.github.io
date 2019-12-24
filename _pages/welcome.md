---
title: "Welcome"
layout: splash
permalink: /
date: 2019-12-23
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/hongkong.jpeg
  actions:
    - label: "Home"
      url: "/home/"
  # caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
excerpt: "Welcome to my website! This website is about my life and learning experience 欢迎来到我的个人网站，我将会在这里分享我的生活以及学习经验"
intro: 
  - excerpt: 'All of belows are my interests, check them out! 这些内容都是我所感兴趣的，快看看吧！'
feature_row:
  - image_path: assets/images/music.jpeg
    # image_caption: "Image courtesy of [Unsplash](https://unsplash.com/)"
    alt: "Music 音乐"
    title: "Music 音乐"
    excerpt: "Guitar and keyboard are my weapon 吉他和键盘是我的武器"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/coding.jpeg
    alt: "IT / AI / ML / DS Skill 技术"
    title: "IT / AI / ML / DS Skill 技术"
    excerpt: "These are my primary jobs 我的首要任务"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/food.jpeg
    alt: "Food 食物"
    title: "Food 食物"
    excerpt: "Delicious food support me 人是铁饭是钢"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row2:
  - image_path: assets/images/travel.jpeg
    # image_caption: "Image courtesy of [Unsplash](https://unsplash.com/)"
    alt: "Travel 旅行"
    title: "Travel 旅行"
    excerpt: "I'm eager to travel 我是多么渴望旅行"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/flight.jpeg
    alt: "Flight 飞行"
    title: "Flight 飞行"
    excerpt: "I'm also a flight enthusiast 同样，我也是一名飞行爱好者"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/exercise.jpeg
    title: "Exercise 锻炼"
    excerpt: "Exercising? HA~ 锻炼？ 哈"
    url: "/home/"
    btn_label: "Read More"
    btn_class: "btn--primary"
# feature_row2:
#   - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
#     alt: "placeholder image 2"
#     title: "Placeholder Image Left Aligned"
#     excerpt: 'This is some sample content that goes here with **Markdown** formatting. Left aligned with `type="left"`'
#     url: "/home/"
#     btn_label: "Read More"
#     btn_class: "btn--primary"
# feature_row3:
#   - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
#     alt: "placeholder image 2"
#     title: "Placeholder Image Right Aligned"
#     excerpt: 'This is some sample content that goes here with **Markdown** formatting. Right aligned with `type="right"`'
#     url: "/home/"
#     btn_label: "Read More"
#     btn_class: "btn--primary"
# feature_row4:
#   - image_path: /assets/images/unsplash-gallery-image-2-th.jpg
#     alt: "placeholder image 2"
#     title: "Placeholder Image Center Aligned"
#     excerpt: 'This is some sample content that goes here with **Markdown** formatting. Centered with `type="center"`'
#     url: "/home/"
#     btn_label: "Read More"
#     btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="feature_row2" type="left" %}

{% include feature_row id="feature_row3" type="right" %}

{% include feature_row id="feature_row4" type="center" %}