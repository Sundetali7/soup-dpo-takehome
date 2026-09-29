"""Synthetic 'messy Russian support tickets' preference dataset with KNOWN injected defects.
Used only to test what soup's data checks catch. Defects -> defects_manifest.json"""
import json, random, os

SEED, N_TRAIN, N_EVAL = 42, 500, 60

CATS = {
    "delivery": {
        "tickets": ["заказ {oid} не пришел уже {d} дней где он??",
                    "Здравствуйте, посылка {oid} застряла, трек не обновляется с {date}",
                    "почему доставка так долго, заказ {oid}, обещали {date}",
                    "курьер не приехал!!! заказ {oid} ждала весь день",
                    "Где мой заказ {oid}? прошло {d} дн. ответьте пожалуйста"],
        "chosen": ["Извините за задержку. Я проверил заказ {oid}: он передан в службу доставки. Чтобы уточнить статус доставки, я открыл запрос перевозчику, ответ придёт в течение 24 часов. Если заказ не придёт до {date}, мы вернём деньги или отправим замену.",
                   "Понимаю, как это неприятно. По заказу {oid} я запросил статус доставки у курьерской службы. Как только получу ответ, напишу вам сюда. Если посылка потеряна, оформим возврат полной суммы."],
    },
    "refund": {
        "tickets": ["хочу вернуть деньги за заказ {oid}, товар не подошел", "Верните деньги!!! {oid} брак",
                    "как оформить возврат? заказ {oid} от {date}",
                    "вернули товар {d} дней назад а деньги так и не пришли, заказ {oid}"],
        "chosen": ["Конечно, помогу с возвратом. Для заказа {oid} я создал заявку на возврат. Деньги вернутся на карту, которой вы платили, обычно за 3–7 рабочих дней. Номер заявки придёт вам на почту.",
                   "Приношу извинения за неудобства. Возврат по заказу {oid} оформлен. Если через 7 рабочих дней деньги не поступят, напишите нам, и мы проверим платёж вместе с банком."],
    },
    "login": {
        "tickets": ["не могу войти в акаунт пишет неверный пароль", "не приходит смс код для входа((",
                    "аккаунт заблокирован почему??? логин {email}", "забыл пароль, почта {email} больше не работает"],
        "chosen": ["Давайте восстановим доступ. Нажмите «Забыли пароль?» на странице входа и укажите {email}. Если письмо не придёт за 10 минут, проверьте папку «Спам» или напишите мне, и я сброшу пароль вручную после проверки личности.",
                   "Помогу восстановить доступ. Я проверил аккаунт {email}: он временно заблокирован после нескольких неудачных попыток входа. Блокировка снимется через 30 минут, после этого войдите через «Забыли пароль?»."],
    },
    "double_charge": {
        "tickets": ["с карты списали два раза за заказ {oid}!!", "двойное списание {sum} руб, заказ {oid}, верните",
                    "Почему деньги сняли дважды? {oid}"],
        "chosen": ["Извините, это неприятная ситуация. Я вижу по заказу {oid} повторное списание {sum} ₽. Второй платёж уже отменён, деньги вернутся на карту в течение 3–5 рабочих дней.",
                   "Проверил: по заказу {oid} действительно прошло повторное списание. Я передал его в платёжный отдел для отмены, сумма {sum} ₽ вернётся на карту. Подтверждение придёт на почту."],
    },
    "app_crash": {
        "tickets": ["приложение вылетает при оплате, андроид {ver}", "после обновления прила не открывается вообще",
                    "app crashes when I open корзина, iphone"],
        "chosen": ["Спасибо, что сообщили. Пожалуйста, обновите приложение до последней версии и перезапустите телефон. Если сбой повторится, пришлите модель устройства и время ошибки, я передам это разработчикам.",
                   "Извините за неудобства. Мы знаем о сбое на Android {ver}. Пожалуйста, обновите приложение: в новой версии ошибка исправлена. Если не поможет, очистите кэш приложения в настройках телефона."],
    },
    "cancel_sub": {
        "tickets": ["как отменить подписку?? деньги снимают каждый месяц", "отпишите меня от премиума, {email}",
                    "хочу отменить автопродление"],
        "chosen": ["Отменить подписку можно в разделе «Профиль → Подписка → Отменить». Доступ сохранится до конца оплаченного периода, новых списаний не будет. Если хотите, я могу отменить её за вас прямо сейчас.",
                   "Готово: я отменил подписку и автопродление для {email}. Премиум будет работать до конца текущего периода, после этого списаний не будет."],
    },
}

REJECT_STYLES = [
    "Ждите.", "Это не наша проблема, обращайтесь в банк.", "Не знаю.",
    "Мы ничего не можем сделать, правила есть правила. Читайте условия на сайте, там всё написано, и не пишите сюда больше по этому вопросу, пожалуйста, у нас очень много обращений и мы не успеваем отвечать всем подряд.",
    "Hello! Please contact support via email.",
    "Спасибо за обращение! Кстати, у нас сейчас скидка 20% на все товары для дома, успейте купить!",
    "Попробуйте переустановить Windows, обычно это помогает в таких случаях.",
    "ваш вопрос очень важен для нас оставайтесь на линии",
]


def noise(s, rng):
    if rng.random() < 0.3: s = s.lower()
    if rng.random() < 0.15: s = s.upper()
    if rng.random() < 0.25:
        i = rng.randrange(1, max(2, len(s) - 2)); s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if rng.random() < 0.2: s = s.replace(",", "").replace(".", "")
    if rng.random() < 0.15: s += rng.choice([" 😡", " 🙏", " ...", " !!!"])
    if rng.random() < 0.1: s = "   " + s + "\n\n"
    return s


def make_row(rng, cat):
    c = CATS[cat]
    ctx = dict(oid=f"#{rng.randint(100000, 999999)}", d=rng.randint(2, 21),
               date=f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}",
               email=f"user{rng.randint(1, 999)}@mail.ru",
               sum=rng.choice([499, 1290, 2490, 5990]), ver=rng.choice([10, 11, 12, 13, 14]))
    prompt = noise(rng.choice(c["tickets"]).format(**ctx), rng)
    chosen = rng.choice(c["chosen"]).format(**ctx)
    if rng.random() < 0.3:
        other = rng.choice([k for k in CATS if k != cat])
        rejected = rng.choice(CATS[other]["chosen"]).format(**ctx)
    else:
        rejected = rng.choice(REJECT_STYLES)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main(out_dir="data/synth"):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED); cats = list(CATS)
    rows = [make_row(rng, cats[i % len(cats)]) for i in range(N_TRAIN)]
    for i, r in enumerate(rows): r["id"] = f"tr{i:04d}"
    defects = {"chosen_eq_rejected": [], "label_swapped": [], "exact_duplicate": [],
               "overlong_prompt": [], "whitespace_only_rejected": []}
    idx = list(range(N_TRAIN)); rng.shuffle(idx)
    take = lambda n: [idx.pop() for _ in range(n)]
    for i in take(15):
        rows[i]["rejected"] = rows[i]["chosen"]; defects["chosen_eq_rejected"].append(rows[i]["id"])
    for i in take(25):
        rows[i]["chosen"], rows[i]["rejected"] = rows[i]["rejected"], rows[i]["chosen"]
        defects["label_swapped"].append(rows[i]["id"])
    for i in take(10):
        rows[i]["prompt"] += " Предыстория: " + "я уже писал вам раньше и никто не ответил. " * 60
        defects["overlong_prompt"].append(rows[i]["id"])
    for i in take(5):
        rows[i]["rejected"] = "   "; defects["whitespace_only_rejected"].append(rows[i]["id"])
    for i in take(10):
        d = dict(rows[i]); d["id"] = rows[i]["id"] + "_dup"; rows.append(d)
        defects["exact_duplicate"].append(d["id"])
    rng.shuffle(rows)
    with open(f"{out_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("prompt", "chosen", "rejected")}, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/train_with_ids.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(defects, open(f"{out_dir}/defects_manifest.json", "w"), indent=2)
    print(f"synthetic rows: {len(rows)} ->", out_dir)
    print({k: len(v) for k, v in defects.items()})


if __name__ == "__main__":
    main()
